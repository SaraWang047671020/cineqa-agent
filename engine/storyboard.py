import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from config.settings import settings
from telemetry.tracer import tracer
from google import genai
from google.genai.types import GenerateContentConfig, ImageConfig

DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"

def generate_storyboard(
    prompt: str,
    negative_prompt: str = "",
    base_image_path: Optional[str] = None,
    reference_images: Optional[dict[str, str]] = None,
    aspect_ratio: str = "16:9",
    use_live_imagen: bool = True,
    model_name: str = "gemini-2.5-flash-image",
    out_dir: str = "temp_eval/storyboards",
    is_first_frame: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generates a pre-flight opening first frame or keyframe using Gemini 2.5 Flash Image.
    """
    out_path_dir = Path(os.path.abspath(out_dir))
    out_path_dir.mkdir(parents=True, exist_ok=True)
    
    # Safe prompt resolution (handles dict or None gracefully)
    if isinstance(prompt, dict):
        clean_prompt = str(prompt.get("final_prompt") or prompt.get("prompt") or prompt)
    elif prompt is None:
        clean_prompt = ""
    else:
        clean_prompt = str(prompt)

    is_ff = kwargs.get("is_first_frame", is_first_frame)

    import uuid
    timestamp = int(time.time())
    file_prefix = "first_frame" if (is_ff and not base_image_path) else "keyframe"
    local_out_path = str(out_path_dir / f"{file_prefix}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg")
    error_msg = None

    with tracer.start_as_current_span("Storyboard.generate"):
        if use_live_imagen:
            try:
                frame_type_label = "FIRST FRAME (t=0.0s)" if (is_ff and not base_image_path) else "KEYFRAME"
                print(f"[CineQA Storyboard] Launching live {frame_type_label} generation via {model_name}...")
                
                # Gemini 3 Pro Image is hosted in the 'global' region, while 2.5 Flash is in us-central1
                loc_override = "global"
                client = settings.get_genai_client(location_override=loc_override)
                
                full_prompt = clean_prompt
                if is_ff and not base_image_path:
                    first_frame_directive = (
                        "[CINEMATOGRAPHIC DIRECTIVE — OPENING FIRST FRAME (t=0.0s)]:\n"
                        "You are generating the EXACT OPENING FIRST FRAME (Frame 0, starting state) of this cinematic shot for an Image-to-Video engine.\n"
                        "- Render the INITIAL STARTING POSE and position of all characters and elements right before the action commences.\n"
                        "- Establish the opening camera framing, composition, depth of field, perspective, and lighting atmosphere.\n"
                        "- Keep the subject in a clean, poised starting state ready to animate (do NOT render mid-action motion blur or post-action aftermath).\n"
                        "- Crisp, photorealistic detail with sharp edges so the video model can seamlessly anchor and animate continuous motion forward from t=0.0s.\n\n"
                        "SCENE DESCRIPTION:\n"
                    )
                    full_prompt = first_frame_directive + clean_prompt

                if negative_prompt:
                    full_prompt = f"{full_prompt}\n\nDo not include: {negative_prompt}"

                config = GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=ImageConfig(aspect_ratio=aspect_ratio)
                )
                
                contents = [full_prompt]
                from google.genai.types import Part
                
                # Base Image (I2I) has highest priority. If we are doing I2I (Last Frame),
                # DO NOT mix in other reference images, as multiple conflicting images confuse Gemini 2.5 Flash Image.
                if base_image_path and os.path.exists(base_image_path):
                    with open(base_image_path, "rb") as f:
                        img_bytes = f.read()
                    mime_type = "image/png" if base_image_path.lower().endswith(".png") else "image/jpeg"
                    contents.insert(0, Part.from_bytes(data=img_bytes, mime_type=mime_type))
                    print(f"[CineQA Keyframe] Using base image for I2I: {base_image_path}. Ignoring other references to prevent artifacting.")
                    # Point 4: Add DIRECTOR'S NOTE for I2I
                    i2i_note = "\n[DIRECTOR\'S NOTE: The attached image is the FIRST FRAME of this shot. Generate the LAST FRAME. CRITICAL CONSISTENCY: You MUST use the EXACT SAME design, textures, colors, and background environment as the provided image. Do NOT re-design the subjects or props based on text descriptions. Just change their position. If an object moved, completely ERASE it from its original starting position (leave empty background). Do NOT clone objects. There is only ONE instance of the moving subject in the final image.]\n"
                    contents.insert(1, i2i_note)
                elif reference_images:
                    ref_parts = []
                    for role, ref_path in reference_images.items():
                        if os.path.exists(ref_path):
                            with open(ref_path, "rb") as f:
                                img_bytes = f.read()
                            mime_type = "image/png" if ref_path.lower().endswith(".png") else "image/jpeg"
                            ref_parts.append(f"\n[{role.upper()} REFERENCE]:")
                            ref_parts.append(Part.from_bytes(data=img_bytes, mime_type=mime_type))
                            print(f"[CineQA Keyframe] Using explicit {role} reference: {ref_path}")
                    if ref_parts:
                        ref_parts.append("\n[DIRECTOR\'S NOTE: The images above are strict DESIGN references for the character, scene, or objects. Do not blend them. CRITICAL: You MUST freely adapt the character\'s pose, facing direction, and camera angle to match the text prompt below. Maintain their clothing/colors/identity, but rotate them in 3D space as needed.]\n")
                        contents = ref_parts + contents
                    pass

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=contents,
                            config=config
                        )
                        
                        image_bytes = None
                        
                        # Safe iteration over response
                        if response.candidates and response.candidates[0].content:
                            parts = response.candidates[0].content.parts
                            if parts:
                                for part in parts:
                                    if getattr(part, "inline_data", None):
                                        image_bytes = part.inline_data.data
                                        break
                        
                        # Check alternative generated_images attribute (SDK variations)
                        if not image_bytes and hasattr(response, "generated_images") and response.generated_images:
                            image_bytes = response.generated_images[0].image.image_bytes
                        
                        if not image_bytes:
                            raise ValueError(f"No image data found in response. Response: {response}")
                                
                        if image_bytes:
                            if isinstance(image_bytes, str):
                                import base64
                                try:
                                    image_bytes = base64.b64decode(image_bytes)
                                except Exception:
                                    image_bytes = image_bytes.encode("utf-8")
                            with open(local_out_path, "wb") as f:
                                f.write(image_bytes)
                            print(f"[CineQA Keyframe] Saved to {local_out_path}")
                            return {
                                "status": "SUCCESS",
                                "image_path": local_out_path,
                                "model_used": model_name,
                                "is_mock": False
                            }
                    except Exception as exc:
                        err_str = str(exc).lower()
                        is_transient = any(k in err_str for k in [
                            "transport", "nameresolutionerror", "11001", "getaddrinfo",
                            "timeout", "connection", "remotedisconnected", "temporarily unavailable",
                            "resourceexhausted", "429", "503", "quota"
                        ])
                        if attempt < max_retries - 1 and is_transient:
                            wait_s = 2.0 * (attempt + 1)
                            print(f"[CineQA Keyframe] Transient issue on attempt {attempt+1} ({exc}), retrying in {wait_s}s...")
                            time.sleep(wait_s)
                        else:
                            raise exc
            except Exception as e:
                error_msg = str(e)
                print(f"[CineQA Keyframe] Exception: {e}. Falling back to mock storyboard...")

        # Fallback to local synth
        create_mock_storyboard(local_out_path, prompt)
        return {
            "status": "SUCCESS",
            "image_path": local_out_path,
            "model_used": "Mock_Image",
            "is_mock": True,
            "error": error_msg
        }

def create_mock_storyboard(out_path: str, prompt: str) -> str:
    """Helper to synthesize fallback image if needed."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (1280, 720), color = (73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((100, 360), f"Keyframe (Simulated)\nPrompt: {prompt[:50]}...", fill=(255, 255, 0))
        img.save(out_path)
    except Exception:
        pass
    return out_path
