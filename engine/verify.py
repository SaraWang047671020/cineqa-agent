"""Frame Verification Engine: Evaluates atomic claims against video frames & reference concept art.
Integrates MAPIE 1.5.0 Conformal Decision Layer for mathematically sound verdict determination.
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from config.settings import settings
from agents.conformal_judge import ConformalJudge

VALID_VERDICTS = {"MATCH", "MISMATCH", "CANNOT_DETERMINE"}

VERIFY_PROMPT = (
    'Claim to verify: "{claim_text}"\n\n'
    "The frames are attached in chronological order. (If reference storyboard/concept art images are provided, they are labeled as REFERENCE_IMAGE_X).\n\n"
    "Judge whether this claim is true, following these steps in order:\n\n"
    "1. Break the claim down into every specific sub-fact that must individually hold true.\n"
    "2. If this claim involves matching a character appearance or storyboard layout from reference images, explicitly compare the video frames against the reference image anchors.\n"
    "3. Left/right judgments must ALWAYS be defined from the VIEWER's perspective looking at the screen.\n"
    "4. Check for causal order consistency: cause must lead forward into effect.\n"
    "5. Output verdict (MATCH / MISMATCH / CANNOT_DETERMINE) with observed evidence."
)

VERIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "checkable_components": {
            "type": "array",
            "items": {"type": "string"},
            "description": "the specific sub-facts that must each individually hold true",
        },
        "frame_observations": {
            "type": "string",
            "description": "what is actually observed across the frames for each sub-fact",
        },
        "all_required_subjects_fully_visible": {"type": "boolean"},
        "artifacts_affect_judgment": {"type": "boolean"},
        "event_causal_order": {"type": "string"},
        "concept_art_consistency": {
            "type": "string",
            "description": "Analysis of character/storyboard consistency against reference art (if applicable)"
        },
        "verdict": {"type": "string", "enum": ["MATCH", "MISMATCH", "CANNOT_DETERMINE"]},
        "observed": {"type": "string", "description": "concise final summary of what you saw"},
        "confidence": {
            "type": "number",
            "description": "Self-reported confidence on a 0.0 to 1.0 scale",
        },
    },
    "required": [
        "checkable_components", "frame_observations",
        "all_required_subjects_fully_visible", "artifacts_affect_judgment",
        "event_causal_order", "verdict", "observed", "confidence",
    ],
}

# Instantiate singleton Conformal Judge calibrated on 92 benchmark rows
_conformal_judge = ConformalJudge(confidence_level=0.80)

def get_clip_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 4.0
    return float(result.stdout.strip())

def extract_frames(video_path: str, out_dir: str, temporal: str, claim_id: str) -> List[Path]:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = get_clip_duration(str(video_path))

    if temporal == "static":
        timestamps = [duration * 0.1, duration * 0.5, duration * 0.9]
    else:
        step = 0.4
        n = max(1, int(duration / step))
        timestamps = [min(i * step, duration - 0.05) for i in range(n)]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"{claim_id}_f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            capture_output=True,
        )
        if out_path.exists():
            frame_paths.append(out_path)
    return frame_paths

def call_gemini_verify(
    claim_text: str, 
    frame_paths: List[Path], 
    reference_image_paths: Optional[Union[str, List[str]]] = None,
    reference_image_path: Optional[str] = None,
    dry_run: bool = False, 
    project: Optional[str] = None, 
    location: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    if dry_run:
        return {
            "verdict": "MATCH",
            "observed": "[Dry-run simulated verification]",
            "confidence": 0.95,
            "checkable_components": ["component_1"],
            "frame_observations": "Simulation observed consistent attributes.",
            "all_required_subjects_fully_visible": True,
            "artifacts_affect_judgment": False,
            "event_causal_order": "consistent"
        }

    from google import genai
    from google.genai import types

    project = project or settings.GOOGLE_CLOUD_PROJECT
    location = location or settings.GOOGLE_CLOUD_LOCATION
    model_name = settings.DEFAULT_GEMINI_MODEL

    client = genai.Client(vertexai=True, project=project, location=location)

    image_list = []
    if reference_image_paths:
        if isinstance(reference_image_paths, str):
            image_list.append(reference_image_paths)
        elif isinstance(reference_image_paths, list):
            image_list.extend(reference_image_paths)
    if reference_image_path and reference_image_path not in image_list:
        image_list.append(reference_image_path)

    contents = []
    for idx, img_path in enumerate(image_list):
        if img_path and os.path.exists(img_path):
            ext = Path(img_path).suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            with open(img_path, "rb") as fh:
                contents.append(
                    types.Part.from_bytes(data=fh.read(), mime_type=mime)
                )
            contents.append(f"[REFERENCE_IMAGE_{idx+1}: {Path(img_path).name}]")

    for p in frame_paths:
        with open(p, "rb") as fh:
            contents.append(
                types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg")
            )

    contents.append(VERIFY_PROMPT.format(claim_text=claim_text))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VERIFY_RESPONSE_SCHEMA,
            temperature=0.1
        ),
    )
    return json.loads(response.text)

CONSENSUS_ROUNDS = 3

def call_gemini_verify_with_consensus(
    claim_text: str, 
    frame_paths: List[Path], 
    reference_image_paths: Optional[Union[str, List[str]]] = None,
    reference_image_path: Optional[str] = None,
    dry_run: bool = False, 
    project: Optional[str] = None, 
    location: Optional[str] = None,
    consensus_rounds: int = CONSENSUS_ROUNDS,
    **kwargs
) -> Dict[str, Any]:
    """
    Executes consensus verification and passes the resulting vote distribution
    into the MAPIE 1.5.0 Conformal Decision Layer for guaranteed risk control.
    """
    first = call_gemini_verify(
        claim_text, frame_paths, 
        reference_image_paths=reference_image_paths, 
        reference_image_path=reference_image_path,
        dry_run=dry_run, project=project, location=location
    )

    if dry_run:
        result = dict(first)
        result["consensus_calls"] = 1
        result["consensus_votes"] = ["MATCH"]
        result["prediction_set"] = ["MATCH"]
        result["conformal_autonomous"] = True
        return result

    votes = [first]
    for _ in range(consensus_rounds - 1):
        votes.append(
            call_gemini_verify(
                claim_text, frame_paths, 
                reference_image_paths=reference_image_paths, 
                reference_image_path=reference_image_path,
                dry_run=dry_run, project=project, location=location
            )
        )

    verdict_strings = [v.get("verdict", "CANNOT_DETERMINE") for v in votes]
    conformal_result = _conformal_judge.evaluate_verdict(verdict_strings)
    
    calibrated_verdict = conformal_result["verdict"]
    representative = next((v for v in votes if v.get("verdict") == calibrated_verdict), votes[0])
    
    result = dict(representative)
    result["verdict"] = calibrated_verdict
    result["consensus_calls"] = len(votes)
    result["consensus_votes"] = verdict_strings
    result["prediction_set"] = conformal_result.get("prediction_set", [calibrated_verdict])
    result["conformal_set_size"] = conformal_result.get("set_size", 1)
    result["conformal_autonomous"] = conformal_result.get("is_autonomous", True)
    result["coverage_guarantee"] = conformal_result.get("coverage_guarantee", 0.80)
    
    if not conformal_result["is_autonomous"]:
        result["observed"] += f" [MAPIE Risk Control: Ambiguous Prediction Set {result['prediction_set']}]"
        
    return result
