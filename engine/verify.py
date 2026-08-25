"""Frame Verification Engine: Evaluates atomic claims against video frames & reference concept art.
Implements 5-Step Mandatory Verification Protocol with high-density adaptive sampling,
continuous shot / no-cut disambiguation, fluid dynamics verification, and MAPIE 1.5.0 Conformal Decision Layer.
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
    "The video frames are attached in chronological order. "
    "(If reference storyboard/character concept art images are provided, they are labeled as REFERENCE_IMAGE_X).\n\n"
    "You MUST execute the following 5-Step Mandatory Verification Protocol in exact sequential order before determining your final verdict:\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "STEP 1: SUB-FACT DECOMPOSITION & PHYSICAL/CINEMATIC DYNAMICS PRE-REGISTRATION\n"
    "• Break down the claim into atomic sub-facts that must each individually hold true.\n"
    "• Specific Action/Motion Verification (CRITICAL): If the claim describes an action (e.g., 'kicked away', 'drawing sword', 'dashing'), you MUST explicitly confirm that the kinetic movement, contact, and displacement corresponding to that exact action happens. If subjects simply stand still, pose, or merely exist near each other without executing the action, mark as MISMATCH.\n"
    "• Single Continuous Shot / No-Cut Claims (CRITICAL): The attached frames are DISCRETE SAMPLES taken at intervals from a single video take. Frame-to-frame position jumps or rapid camera pans between sampled frames are normal sampling intervals, NOT video cuts! An actual 'CUT' only occurs if there is an abrupt, total discontinuity in environment/scene (e.g. instant teleportation from outdoor roof to indoor room, or instant camera perspective teleportation with completely disconnected geometry). If lighting, environment, and subject motion remain topologically continuous across the timeline, judge as a continuous shot (MATCH).\n"
    "• Fluid & Liquid Dynamics: Liquid flow, dripping, or bleeding requires ACTIVE, CONTINUOUS DOWNWARD DISPLACEMENT of liquid across sequential frames under gravity. Static blood stains, painted red streaks, or stationary droplets that remain frozen in position do NOT constitute 'flowing/dripping' -> mark as fluid physics failure.\n"
    "• Wearing/Holding relations: Check for physical contact/attachment. Verify there is no mesh clipping (e.g. blade clipping through palm instead of being gripped).\n"
    "• Temporal changes: Explicitly compare early frames (t0) vs late frames (t_end); check intermediate frames for transient micro-events.\n"
    "• Causal events (collision, fracture, ignition, transformation): Pre-register the canonical causal timeline (Cause -> Intermediate -> Effect).\n"
    "• Count claims: Count only prominent, salient entity instances. Ignore distant background noise pixels, but count distinct intended scene elements.\n"
    "• Multimodal Reference Art: If REFERENCE_IMAGE is present, anchor character hair/attire/layout against the reference art.\n\n"
    "STEP 2: OBJECTIVE OBSERVATION & COORDINATE ANCHORING (NO PREMATURE CONCLUSIONS)\n"
    "• Screen-Space Orientation: Left and Right are ALWAYS defined strictly from the VIEWER\'S PERSPECTIVE looking at the screen (Viewer Left / Viewer Right), NEVER the subject\'s anatomical perspective.\n"
    "• Horizontal Movement Anchoring: For motion claims, explicitly record the subject\'s horizontal position in the first frame vs final frame (e.g. Left Edge X:15% -> Right Edge X:85%).\n"
    "• Camera Motion Disentanglement: Distinguish between camera panning/tracking vs subject locomotion in the scene.\n\n"
    "STEP 3: EVIDENCE SUFFICIENCY, INEQUALITY BOUNDING & INCIDENTAL ACTORS\n"
    "• Partial Visibility: A full 100% silhouette outline is not required if the specific relation/attribute is unmistakable.\n"
    "• Incidental Actor Tolerance: If the target entity and its verified action hold true in the primary action window, late-frame peripheral appearances (e.g. incidental hand entering frame, background ambient motion) do NOT invalidate the claim unless the claim explicitly specified 'in complete isolation' or 'no hands'.\n"
    "• Lower-Bound Claims ('at least N times bigger/taller'): If the cropped/partially visible entity is the one required to be larger and its visible portion already exceeds the threshold, unobserved portions only increase the true size -> counts as SUFFICIENT evidence.\n"
    "• Depth Perspective Check: Ensure entities being compared reside on the same focal/depth plane to rule out foreshortening/perspective distortions.\n\n"
    "STEP 4: LOCALIZED AI ARTIFACT DISCRIMINATION\n"
    "• Set `artifacts_affect_judgment=true` ONLY if AI generation defects (morphing, limb fusion, blur) occur directly in the spatial-temporal ROI needed to verify this claim.\n"
    "• If the core sub-fact is clearly verifiable in clean frames/regions, unrelated background artifacts do NOT invalidate the judgment.\n\n"
    "STEP 5: VERDICT FORMULATION & REVERSE-DIFFUSION CAUSAL CHECK\n"
    "• Output MATCH if and only if ALL sub-facts hold true with verified physical motion, topological continuity, and forward causality.\n"
    "• Output MISMATCH if any sub-fact is contradicted (e.g. hard scene cut, static blood texture instead of active flowing liquid), OR if a causal event displays reverse motion/un-breaking (e.g. debris assembling back into an intact object instead of shattering).\n"
    "• Output CANNOT_DETERMINE only if evidence is genuinely insufficient or localized artifacts obstruct evaluation. Do not abuse abstention out of generic caution.\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

VERIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "checkable_components": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The specific atomic sub-facts that must each individually hold true",
        },
        "frame_observations": {
            "type": "string",
            "description": "Objective observations across frames with viewer-perspective coordinate tracking and topological continuity analysis",
        },
        "kinetic_motion_and_context": {
            "type": "string",
            "description": "Evaluate the action based on visible motion, contact, OR strong contextual cues (e.g., motion blur, physical aftermath, flying debris). Acknowledge that generative videos may not have perfect frame-by-frame physics."
        },
        "evidence_sufficiency_check": {
            "type": "string",
            "description": "Balance your judgment. Do not demand irrefutable proof. If the action is reasonably depicted, visually implied, or present without glaring contradictions, lean towards acceptance. Only mark MISMATCH if the action is entirely missing, completely stationary, or egregiously contradicts the claim."
        },
        "all_required_subjects_fully_visible": {"type": "boolean"},
        "artifacts_affect_judgment": {
            "type": "boolean",
            "description": "True only if localized artifacts directly obstruct verification of this specific claim"
        },
        "event_causal_order": {
            "type": "string",
            "description": "Detailed analysis of forward causality (Cause -> Effect) vs reverse motion check"
        },
        "concept_art_consistency": {
            "type": "string",
            "description": "Analysis of character/storyboard feature alignment against reference images (if present)"
        },
        "verdict": {"type": "string", "enum": ["MATCH", "MISMATCH", "CANNOT_DETERMINE"]},
        "observed": {"type": "string", "description": "Concise final synthesis of observed visual evidence"},
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
        timestamps = [
            duration * 0.08, 
            duration * 0.28, 
            duration * 0.50, 
            duration * 0.72, 
            duration * 0.92
        ]
    else:
        step = 0.20
        n = max(1, min(20, int(duration / step)))
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
            "checkable_components": ["atomic_component_1"],
            "frame_observations": "Simulation observed consistent topological continuity without scene cuts.",
            "all_required_subjects_fully_visible": True,
            "artifacts_affect_judgment": False,
            "event_causal_order": "Verified continuous shot without cut transitions."
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
