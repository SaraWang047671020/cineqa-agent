"""Google VEO Video Generation Adapter: Live Google Cloud Vertex AI Video Generation & Closed Loop.
Generates video takes via Google Cloud Vertex AI (Veo 3.1 Fast), exports to GCS, and downloads to local disk.
"""

import os
import re

def _get_resized_image_bytes(image_path: str, target_size=(1280, 720)) -> bytes:
    from PIL import Image
    import io
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.settings import settings
from telemetry.tracer import tracer
from google import genai
from google.genai.types import GenerateVideosConfig
from google.cloud import storage

DEFAULT_VEO_MODEL = "veo-3.1-fast-generate-001"
DEFAULT_GCS_BUCKET = "project-aefe3ba2-ab8b-478a-82d-veo"

_cached_auth_creds = None
_auth_lock = None

def _get_access_token() -> str:
    """
    Acquires a Google Cloud access token with in-memory caching,
    resilient retry on network/DNS glitches, and gcloud CLI fallback.
    """
    global _cached_auth_creds, _auth_lock
    import threading
    if _auth_lock is None:
        _auth_lock = threading.Lock()

    with _auth_lock:
        if _cached_auth_creds and _cached_auth_creds.valid and _cached_auth_creds.token:
            return _cached_auth_creds.token

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import google.auth
        from google.auth.transport.requests import Request

        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        try:
            if _cached_auth_creds is None:
                sa_json = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
                if sa_json:
                    import json
                    from google.oauth2 import service_account
                    sa_info = json.loads(sa_json)
                    creds = service_account.Credentials.from_service_account_info(
                        sa_info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    _cached_auth_creds = creds
                else:
                    creds, _ = google.auth.default()
                    _cached_auth_creds = creds
            else:
                creds = _cached_auth_creds

            if not creds.valid:
                req = Request(session=session)
                creds.refresh(req)

            if creds.token:
                return creds.token
        except Exception as e:
            print(f"[CineQA Auth] google.auth refresh encountered error: {e}. Falling back to gcloud CLI...")

        # Fallback to gcloud CLI
        try:
            import shutil
            import subprocess
            gcloud_cmd = shutil.which("gcloud") or shutil.which("gcloud.cmd")
            if gcloud_cmd:
                res = subprocess.run(
                    [gcloud_cmd, "auth", "print-access-token"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True
                )
                tok = res.stdout.strip()
                if tok:
                    print("[CineQA Auth] Successfully retrieved access token via gcloud CLI fallback.")
                    return tok
        except Exception as e2:
            print(f"[CineQA Auth] gcloud CLI fallback failed: {e2}")

        if _cached_auth_creds and _cached_auth_creds.token:
            return _cached_auth_creds.token

        raise RuntimeError("Failed to obtain a valid Google Cloud access token after retries and gcloud fallback.")

def generate_video(
    prompt: str,
    negative_prompt: str = "", mask_video_path: str = None,
    reference_images: Optional[dict[str, str]] = None,
    style_image_path: Optional[str] = None,
    seed: Optional[int] = None,
    first_frame_path: Optional[str] = None,
    last_frame_path: Optional[str] = None,
    source_video_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 4,
    fps: int = 24,
    use_live_veo: bool = True,
    out_dir: str = "temp_eval/generated_takes",
    video_engine: str = "veo-3.1-fast-generate-001",
    resolution: str = "720p",
    previous_interaction_id: Optional[str] = None,
    interaction_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generates video takes using live Google Vertex AI Veo 3.1, exports to GCS bucket,
    and automatically downloads the .mp4 file to local disk for instant verification & playback.
    """
    # [Camera Movement Fix] Image-to-Image (I2I) interpolation fundamentally restricts camera movement 
    # because the 2D end-frame fixes the vanishing points. If the user explicitly asks for camera movement,
    # we MUST drop the last_frame_path to let Veo generate native 3D camera sweeps.
    camera_keywords = ["pan", "tilt", "zoom", "track", "dolly", "crane", "push in", "pull out"]
    if last_frame_path and any(re.search(r'\b' + kw + r'\b', prompt.lower()) for kw in camera_keywords):
        print(f"[CineQA Veo] Camera movement detected in prompt! Dropping last_frame_path to unlock Veo's native 3D camera routing.")
        last_frame_path = None

    out_path_dir = Path(os.path.abspath(out_dir))
    out_path_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    local_out_path = str(out_path_dir / f"veo_take_remediated_{timestamp}.mp4")
    output_gcs_uri = f"gs://{DEFAULT_GCS_BUCKET}/healed_takes"

    with tracer.start_as_current_span("VeoGenerator.generate_video"):
        if use_live_veo:
            try:
                print(f"[CineQA Veo] Launching live generation via Vertex AI ({DEFAULT_VEO_MODEL})...")
                loc_override = 'us-east5' if 'omni' in video_engine else 'us-central1'
                client = settings.get_genai_client(location_override=loc_override)

                import random
                from google.genai.types import VideoGenerationReferenceImage, Image, Video
                
                # If seed is not provided, generate a random one to record it
                active_seed = seed if seed is not None else random.randint(0, 2147483647)


                if last_frame_path:
                    anti_jumpcut = "jump cut, crossfade, instant transition, sudden snapping, time-lapse, glitch"
                    negative_prompt = (negative_prompt + ", " + anti_jumpcut) if negative_prompt else anti_jumpcut
                config_kwargs = {
                    "aspect_ratio": aspect_ratio,
                    "duration_seconds": str(duration_seconds),
                    "number_of_videos": 1,
                    "output_gcs_uri": output_gcs_uri,
                    "negative_prompt": negative_prompt,
                    "seed": active_seed,
                    "resolution": "1080p"
                }
                
                # Handle explicitly designated ASSET and STYLE references
                ref_images_payload = []
                
                # ASSET References (up to 3)
                if reference_images:
                    config_kwargs["duration_seconds"] = "8" # Image-to-video ONLY supports 8 seconds in Veo 3.1
                    count = 0
                    for role, img_path in reference_images.items():
                        if count >= 3: break
                        if os.path.exists(img_path):
                            with open(img_path, "rb") as f:
                                img_bytes = f.read()
                            mime_type = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
                            ref_images_payload.append(
                                VideoGenerationReferenceImage(
                                    image=Image(image_bytes=img_bytes, mime_type=mime_type),
                                    reference_type="ASSET"
                                )
                            )
                            count += 1
                
                # STYLE Reference (up to 1)
                if style_image_path and os.path.exists(style_image_path):
                    config_kwargs["duration_seconds"] = "8"
                    with open(style_image_path, "rb") as f:
                        img_bytes = f.read()
                    mime_type = "image/png" if style_image_path.lower().endswith(".png") else "image/jpeg"
                    ref_images_payload.append(
                        VideoGenerationReferenceImage(
                            image=Image(image_bytes=img_bytes, mime_type=mime_type),
                            reference_type="STYLE"
                        )
                    )
                
                if first_frame_path and os.path.exists(first_frame_path):
                    print("[CineQA Veo] I2V mode active. Clearing ASSET/STYLE references to avoid API conflict (Image and reference images cannot be both set).")
                    ref_images_payload = []
                    config_kwargs.pop("reference_images", None)

                if ref_images_payload:
                    config_kwargs["reference_images"] = ref_images_payload
                    print(f"[CineQA Veo] Using {len(ref_images_payload)} reference image(s) (ASSET/STYLE)")

                # Handle Last Frame (for Frame Interpolation)
                if last_frame_path and os.path.exists(last_frame_path):
                    img_bytes = _get_resized_image_bytes(last_frame_path)
                    mime_type = "image/jpeg"
                    config_kwargs["last_frame"] = Image(image_bytes=img_bytes, mime_type=mime_type)
                    print(f"[CineQA Veo] Using last_frame interpolation: {last_frame_path}")

                # Handle Inpaint Mask
                if mask_path and os.path.exists(mask_path):
                    from google.genai.types import VideoGenerationMask
                    with open(mask_path, "rb") as f:
                        img_bytes = f.read()
                    mime_type = "image/png" if mask_path.lower().endswith(".png") else "image/jpeg"
                    config_kwargs["mask"] = VideoGenerationMask(
                        image=Image(image_bytes=img_bytes, mime_type=mime_type),
                        mask_mode="INSERT"
                    )
                    print(f"[CineQA Veo] Using inpaint mask (INSERT mode): {mask_path}")

                config = GenerateVideosConfig(**config_kwargs)
                
                from google.genai.types import GenerateVideosSource
                source_kwargs = {"prompt": prompt}
                
                if source_video_path and os.path.exists(source_video_path):
                    with open(source_video_path, "rb") as f:
                        vid_bytes = f.read()
                    source_kwargs["video"] = Video(video_bytes=vid_bytes, mime_type="video/mp4")
                    print(f"[CineQA Veo] Using VIDEO EXTENSION from {source_video_path}")
                    if "image" in source_kwargs:
                        del source_kwargs["image"]
                if last_frame_path:
                    source_kwargs["prompt"] += f"\n\n[PACING & INTERPOLATION DIRECTIVE: The physical action or state-change (e.g., melting, moving) MUST occur slowly, fluidly, and continuously across the ENTIRE {duration_seconds}-second duration of the clip. Show every micro-step of the transformation. DO NOT perform a simple cross-fade, morph, or blend. NO jump cuts or instant snapping.]"
                
                if first_frame_path and os.path.exists(first_frame_path):
                    img_bytes = _get_resized_image_bytes(first_frame_path)
                    mime_type = "image/jpeg"
                    source_kwargs["image"] = Image(image_bytes=img_bytes, mime_type=mime_type)
                    print(f"[CineQA Veo] Using FIRST FRAME (I2V) from {first_frame_path}")
                
                source = GenerateVideosSource(**source_kwargs)
                if "omni" in video_engine:
                    print(f"[CineQA Veo] Bypassing SDK, using direct REST interactions API for Omni...")
                    import requests
                    import base64
                    
                    access_token = _get_access_token()
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json; charset=utf-8"
                    }
                    inputs = []
                    if first_frame_path and os.path.exists(first_frame_path):
                        with open(first_frame_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                            mime = "image/png" if first_frame_path.lower().endswith(".png") else "image/jpeg"
                        inputs.append({"type": "image", "data": b64, "mime_type": mime})
                    if last_frame_path and os.path.exists(last_frame_path):
                        with open(last_frame_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                            mime = "image/png" if last_frame_path.lower().endswith(".png") else "image/jpeg"
                        inputs.append({"type": "image", "data": b64, "mime_type": mime})
                    if source_video_path and os.path.exists(source_video_path):
                        with open(source_video_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        inputs.append({"type": "video", "data": b64, "mime_type": "video/mp4"})
                        
                    if mask_video_path and os.path.exists(mask_video_path):
                        with open(mask_video_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        # Add mask video as an input hint for Omni V2V inpainting
                        inputs.append({"type": "video", "data": b64, "mime_type": "video/mp4"})
                    
                    if prompt:
                        inputs.append({"type": "text", "text": prompt})
                    
                    # Auto-detect task: Treat provided image as reference image for style & character grounding
                    if source_video_path and os.path.exists(source_video_path):
                        task = "edit"
                    elif first_frame_path and os.path.exists(first_frame_path):
                        # Treat the storyboard keyframe as a reference image for dynamic motion
                        task = "reference_to_video"
                    elif reference_images:
                        task = "reference_to_video"
                    else:
                        task = "text_to_video"

                    # Clamp duration between 3s and 10s per Omni spec
                    clamped_duration = max(3, min(10, duration_seconds))
                    if task == "edit" and source_video_path and os.path.exists(source_video_path):
                        try:
                            import cv2
                            cap = cv2.VideoCapture(source_video_path)
                            fps_val = cap.get(cv2.CAP_PROP_FPS) or 24.0
                            frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (clamped_duration * fps_val)
                            measured_dur = int(round(frame_cnt / fps_val))
                            cap.release()
                            if 3 <= measured_dur <= 10:
                                clamped_duration = measured_dur
                                print(f"[CineQA Omni] Measured source video duration for edit task: {clamped_duration}s")
                        except Exception as e:
                            print(f"[CineQA Omni] Could not measure video duration via cv2: {e}")

                    duration_str = f"{clamped_duration}s"

                    # Build response_format (720p is the standard for Omni Interactions API)
                    # Note: edit task forbids aspect_ratio in response_format per API validation rule
                    resp_format = {
                        "type": "video",
                        "resolution": resolution,
                        "duration": duration_str
                    }
                    if task != "edit":
                        resp_format["aspect_ratio"] = aspect_ratio

                    payload = {
                        "model": video_engine,
                        "input": inputs,
                        "response_format": resp_format,
                        "generation_config": {
                            "video_config": {
                                "task": task
                            }
                        }
                    }

                    target_interaction_id = interaction_id or previous_interaction_id
                    # Google Omni rule: previous_interaction_id is forbidden if generation_config.video_config.task is explicitly set.
                    # When video task (e.g. edit, reference_to_video) is set, passing source_video in `input` is the official way to fine-tune.
                    if target_interaction_id and not task:
                        payload["previous_interaction_id"] = target_interaction_id

                    endpoint = f"https://aiplatform.googleapis.com/v1beta1/projects/{settings.GOOGLE_CLOUD_PROJECT}/locations/global/interactions"
                    print(f"Calling Omni API at {endpoint} with task={task}, duration={duration_str}, resolution={resp_format['resolution']}...")
                    
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry
                    omni_session = requests.Session()
                    omni_retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504], raise_on_status=False)
                    omni_session.mount("https://", HTTPAdapter(max_retries=omni_retries))

                    resp = omni_session.post(endpoint, headers=headers, json=payload, timeout=(30, 300))
                    if resp.status_code != 200:
                        print(f"Omni API Error: {resp.status_code} - {resp.text}")
                        # Fallback for previous_interaction_id if model preview rejects it on this path
                        if "previous_interaction_id" in resp.text:
                            print("[CineQA Omni] Retrying without previous_interaction_id as standalone edit/generation...")
                            payload.pop("previous_interaction_id", None)
                            resp = omni_session.post(endpoint, headers=headers, json=payload, timeout=(30, 300))
                        if resp.status_code != 200:
                            err_msg = resp.text
                            try:
                                err_json = resp.json()
                                err_msg = err_json.get("error", {}).get("message", resp.text)
                            except Exception:
                                pass
                            raise RuntimeError(f"Omni API Error ({resp.status_code}): {err_msg}")
                        
                    import json
                    resp_data = resp.json()
                    with open("omni_response_debug.json", "w") as f:
                        json.dump(resp_data, f)
                    
                    # Recursively find the video data in the response
                    def find_video_data(obj):
                        if isinstance(obj, dict):
                            if obj.get("type") == "video" and "data" in obj:
                                return obj["data"]
                            for v in obj.values():
                                res = find_video_data(v)
                                if res: return res
                        elif isinstance(obj, list):
                            for item in obj:
                                res = find_video_data(item)
                                if res: return res
                        return None
                        
                    video_b64 = find_video_data(resp_data)
                    
                    if not video_b64:
                        raise Exception("Could not find video data in Omni response. Check omni_response_debug.json")
                        
                    with open(local_out_path, "wb") as f:
                        f.write(base64.b64decode(video_b64))
                        
                    interaction_id = resp_data.get("id")
                    print(f"[CineQA Veo] Omni video saved to {local_out_path} (Interaction ID: {interaction_id})")
                    return {
                        "video_path": local_out_path,
                        "model_used": video_engine,
                        "interaction_id": interaction_id
                    }


                else:


                    operation = client.models.generate_videos(


                        model=video_engine,


                        source=source,


                        config=config


                    )

                print(f"[CineQA Veo] Operation started: {operation.name}. Polling Vertex AI...")
                
                # Poll Vertex AI until generation completes
                poll_count = 0
                max_polls = 60 # max 10 mins
                while not operation.done and poll_count < max_polls:
                    poll_count += 1
                    time.sleep(10)
                    operation = client.operations.get(operation)

                if operation.error:
                    raise RuntimeError(f"Vertex AI Veo error: {operation.error}")

                videos = operation.response.generated_videos
                if videos:
                    gcs_uri = videos[0].video.uri
                    print(f"[CineQA Veo] Succeeded: {gcs_uri}. Downloading to {local_out_path}...")
                    
                    # Download from GCS bucket to local disk
                    if gcs_uri.startswith("gs://"):
                        parts = gcs_uri[5:].split("/", 1)
                        bucket_name = parts[0]
                        blob_name = parts[1]
                        
                        storage_client = storage.Client(project=settings.GOOGLE_CLOUD_PROJECT)
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        blob.download_to_filename(local_out_path)
                        
                        if os.path.exists(local_out_path) and os.path.getsize(local_out_path) > 1000:
                            print(f"[CineQA Veo] Download complete: {local_out_path} ({os.path.getsize(local_out_path)} bytes)")
                            return {
                                "status": "SUCCESS",
                                "video_path": os.path.abspath(local_out_path),
                                "model_used": f"{video_engine} (Google Cloud Vertex AI)",
                                "aspect_ratio": aspect_ratio,
                                "duration_seconds": duration_seconds,
                                "mode": "live_vertex_ai_veo",
                                "prompt_applied": prompt,
                                "negative_prompt_applied": negative_prompt,
                                "seed_used": active_seed,
                                "gcs_uri": gcs_uri
                            }
            except Exception as e:
                print(f"[CineQA Veo] Live Veo exception: {e}")
                raise e

def create_bulletproof_sample_clip(out_path: str, duration: int = 5, fps: int = 24) -> str:
    """Helper to synthesize fallback clip if needed."""
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import cv2
        import numpy as np
        width, height = 1280, 720
        total_frames = int(duration * fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if out.isOpened():
            for i in range(total_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 0] = np.linspace(25, 45, width, dtype=np.uint8)
                frame[:, :, 1] = np.linspace(15, 30, width, dtype=np.uint8)
                frame[:, :, 2] = np.linspace(10, 20, width, dtype=np.uint8)
                cv2.putText(frame, "Google VEO - Remediated Take", (width // 2 - 300, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 230, 0), 2, cv2.LINE_AA)
                out.write(frame)
            out.release()
    except Exception:
        pass
    return out_path
