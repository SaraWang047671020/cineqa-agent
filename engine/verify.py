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
from typing import List, Dict, Any, Optional, Union, Tuple
from config.settings import settings
from agents.conformal_judge import ConformalJudge

VALID_VERDICTS = {"MATCH", "MISMATCH", "CANNOT_DETERMINE"}

PROMPT_BASE = """Claim to verify: "{claim_text}"\n\nThe video frames are attached in chronological order, each preceded by a label [FRAME_i @ t=X.XXs] giving its exact timestamp in the take. Use these timestamps to reason about sampling gaps, motion speed, and temporal continuity. (If reference storyboard/character concept art images are provided, they are labeled as REFERENCE_IMAGE_X).\n\nYou MUST execute the following 5-Step Mandatory Verification Protocol in exact sequential order before determining your final verdict:\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSTEP 1: SUB-FACT DECOMPOSITION & PHYSICAL/CINEMATIC DYNAMICS PRE-REGISTRATION\n• Break down the claim into atomic sub-facts that must each individually hold true.\n• Wearing/Holding relations: Check for physical contact/attachment. Verify there is no mesh clipping (e.g. blade clipping through palm instead of being gripped).\nSTEP 2: OBJECTIVE OBSERVATION & COORDINATE ANCHORING (NO PREMATURE CONCLUSIONS)\n• Screen-Space Orientation: Left and Right are ALWAYS defined strictly from the VIEWER'S PERSPECTIVE looking at the screen (Viewer Left / Viewer Right), NEVER the subject's anatomical perspective.\n• Horizontal Movement Anchoring: For motion claims, explicitly record the subject's horizontal position in the first frame vs final frame (e.g. Left Edge X:15% -> Right Edge X:85%).\n• Camera Tracking & Background Reference (CRITICAL): If a subject remains centrally framed across multiple frames (their screen X/Y coordinates do not change), you MUST observe the BACKGROUND (e.g., ground, buildings, bridge). If the background is shifting or scrolling continuously in the opposite direction, the camera is tracking the subject. This PROVES the subject is moving through the world. DO NOT falsely conclude the subject is "not moving" just because they stay in the center of the screen during a tracking shot!\n\nSTEP 3: EVIDENCE SUFFICIENCY, INEQUALITY BOUNDING & INCIDENTAL ACTORS\n• Partial Visibility: A full 100% silhouette outline is not required if the specific relation/attribute is unmistakable.\n• Incidental Actor Tolerance: If the target entity and its verified action hold true in the primary action window, late-frame peripheral appearances (e.g. incidental hand entering frame, background ambient motion) do NOT invalidate the claim unless the claim explicitly specified 'in complete isolation' or 'no hands'.\n• Lower-Bound Claims ('at least N times bigger/taller'): If the cropped/partially visible entity is the one required to be larger and its visible portion already exceeds the threshold, unobserved portions only increase the true size -> counts as SUFFICIENT evidence.\n• Depth Perspective Check: Ensure entities being compared reside on the same focal/depth plane to rule out foreshortening/perspective distortions.\n\nSTEP 4: LOCALIZED AI ARTIFACT DISCRIMINATION\n• Set `artifacts_affect_judgment=true` ONLY if AI generation defects (morphing, limb fusion, blur) occur directly in the spatial-temporal ROI needed to verify this claim.\n• If the core sub-fact is clearly verifiable in clean frames/regions, unrelated background artifacts do NOT invalidate the judgment.\n\nSTEP 5: VERDICT FORMULATION & REVERSE-DIFFUSION CAUSAL CHECK\n• Output MATCH if and only if ALL sub-facts hold true with verified physical motion, topological continuity, and forward causality.\n• Output MISMATCH if any sub-fact is contradicted (e.g. hard scene cut, static blood texture instead of active flowing liquid), OR if a causal event displays reverse motion/un-breaking (e.g. debris assembling back into an intact object instead of shattering).\n• Output CANNOT_DETERMINE only if evidence is genuinely insufficient or localized artifacts obstruct evaluation. Do not abuse abstention out of generic caution.\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

BLOCK_ACTION = """• Implicit Entity Deduction (CRITICAL): You MUST infer unstated physical subjects logically required by the verbs. For example, if the text says 'is kicked', a 'foot/leg' MUST be visibly present to perform the action. If 'is slashed', a 'blade' MUST be present. An object cannot act upon itself. If the implicit agent is never visible AND there is no physical aftermath evidence (motion blur, sudden displacement, impact deformation, flying debris), output CANNOT_DETERMINE for this sub-fact rather than MISMATCH; the action may have occurred off-frame or between sampled frames.\n• Specific Action/Motion Verification (CRITICAL): If the claim describes an action (e.g., 'kicked away', 'drawing sword', 'dashing'), you MUST explicitly confirm that the kinetic movement, contact, and displacement corresponding to that exact action happens. If across ALL sampled frames the subjects only stand still, pose, or coexist with zero positional change and zero aftermath evidence, mark as MISMATCH. If frames show partial evidence of the motion (mid-action pose change, displacement between consecutive frames), treat it as evidence FOR the action."""

BLOCK_SEQUENTIAL = """• Single Continuous Shot / No-Cut Claims (CRITICAL): The attached frames are DISCRETE SAMPLES taken at intervals from a single video take. Frame-to-frame position jumps or rapid camera pans between sampled frames are normal sampling intervals, NOT video cuts! An actual 'CUT' only occurs if there is an abrupt, total discontinuity in environment/scene (e.g. instant teleportation from outdoor roof to indoor room, or instant camera perspective teleportation with completely disconnected geometry). If lighting, environment, and subject motion remain topologically continuous across the timeline, judge as a continuous shot (MATCH).\n• Fluid & Liquid Dynamics: Liquid flow, dripping, or bleeding requires ACTIVE, CONTINUOUS DOWNWARD DISPLACEMENT of liquid across sequential frames under gravity. Static blood stains, painted red streaks, or stationary droplets that remain frozen in position do NOT constitute 'flowing/dripping' -> mark as fluid physics failure.\n• Temporal changes: Explicitly compare early frames (t0) vs late frames (t_end); check intermediate frames for transient micro-events.\n• Causal events (collision, fracture, ignition, transformation): Pre-register the canonical causal timeline (Cause -> Intermediate -> Effect)."""

BLOCK_COUNT = """• Count claims: Count only prominent, salient entity instances. Ignore distant background noise pixels, but count distinct intended scene elements."""

BLOCK_REFERENCE = """• Multimodal Reference Art: If REFERENCE_IMAGE is present, anchor character hair/attire/layout against the reference art."""

def build_verify_prompt(claim_text: str, claim_type: str, temporal: str, has_reference: bool) -> str:
    blocks = [PROMPT_BASE.format(claim_text=claim_text)]
    if claim_type == "action":
        blocks.insert(1, BLOCK_ACTION)
    if temporal == "sequential":
        blocks.insert(1, BLOCK_SEQUENTIAL)
    if claim_type == "count":
        blocks.insert(1, BLOCK_COUNT)
    if has_reference:
        blocks.insert(1, BLOCK_REFERENCE)
    return "\n".join(blocks)

VERIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "checkable_components": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The specific atomic sub-facts that must each individually hold true",
        },
        "entity_presence_check": {
            "type": "string",
            "description": "Step 1 (Objects & Implicit Entities): Identify all explicit nouns. THEN, logically deduce any unstated physical entities required by the verbs (e.g., 'kicked' requires a 'foot'). Confirm exactly which of these explicit and implicit entities are visibly present."
        },
        "action_execution_check": {
            "type": "string",
            "description": "Step 2 (Verbs): Identify all actions (e.g., kicked, broken, swinging). For the entities found in Step 1, verify if these specific actions actually occurred between them."
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
            "description": "For each sub-fact, state neutrally: (a) which frames provide supporting evidence, (b) which frames provide contradicting evidence, (c) whether the evidence is SUFFICIENT, PARTIAL, or ABSENT. Do not apply leniency or strictness here; report evidence only."
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
        "checkable_components", "entity_presence_check", "action_execution_check",
        "frame_observations", "kinetic_motion_and_context", "evidence_sufficiency_check",
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

def extract_frames(video_path: str, out_dir_str: str, temporal: str, claim_id: str, sampling_strategy: str = "uniform") -> List[Tuple[str, float]]:
    """Extracts frames from video based on temporal requirement and semantic sampling strategy."""
    video = Path(video_path)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Fast check duration using ffprobe
    duration = 4.0
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True
        )
        if res.stdout.strip():
            duration = float(res.stdout.strip())
    except Exception:
        pass

    if temporal == "static":
        n = 3
        timestamps = [duration * 0.1, duration * 0.5, duration * 0.9]
    else:
        if sampling_strategy == "fast_burst":
            # High density sampling: focus on the first 1.5 seconds for instant/fast actions
            burst_duration = min(duration, 1.5)
            n = min(20, max(5, int(burst_duration * 12))) # Approx 12 fps burst
            start = duration * 0.02
            end = max(burst_duration, duration * 0.02)
            timestamps = [start + (end - start) * i / max(1, n - 1) for i in range(n)]
        else:
            # Uniform sequential sampling across the whole video
            n = max(2, min(20, int(duration / 0.20)))
            start, end = duration * 0.02, max(duration - 0.05, duration * 0.02)
            timestamps = [start + (end - start) * i / max(1, n - 1) for i in range(n)]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"{claim_id}_f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(out_path)],
            capture_output=True,
        )
        if out_path.exists():
            frame_paths.append((str(out_path), ts))
    return frame_paths

def call_gemini_verify(
    claim_text: str, 
    frame_paths: List[tuple], 
    reference_image_paths: Optional[Union[str, List[str]]] = None,
    reference_image_path: Optional[str] = None,
    dry_run: bool = False, 
    project: Optional[str] = None, 
    location: Optional[str] = None,
    temperature: float = 0.1,
    claim_type: str = "state",
    temporal: str = "static",
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

    for i, (p, ts) in enumerate(frame_paths):
        contents.append(f"[FRAME_{i} @ t={ts:.2f}s]")
        with open(p, "rb") as fh:
            contents.append(
                types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg")
            )

    contents.append(build_verify_prompt(claim_text, claim_type, temporal, bool(image_list)))

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VERIFY_RESPONSE_SCHEMA,
                temperature=temperature
            ),
        )
        result = json.loads(response.text)
    except Exception as e:
        return {
            "verdict": "CANNOT_DETERMINE",
            "observed": f"[Verification call failed: {type(e).__name__}: {e}]",
            "confidence": 0.0,
            "checkable_components": [],
            "frame_observations": "",
            "all_required_subjects_fully_visible": False,
            "artifacts_affect_judgment": False,
            "event_causal_order": "",
        }
        
    VALID_VERDICTS = {"MATCH", "MISMATCH", "CANNOT_DETERMINE"}
    if result.get("verdict") not in VALID_VERDICTS:
        result["verdict"] = "CANNOT_DETERMINE"
    return result

CONSENSUS_ROUNDS = 3

def call_gemini_verify_with_consensus(
    claim_text: str, 
    frame_paths: List[tuple], 
    reference_image_paths: Optional[Union[str, List[str]]] = None,
    reference_image_path: Optional[str] = None,
    dry_run: bool = False, 
    project: Optional[str] = None, 
    location: Optional[str] = None,
    consensus_rounds: int = CONSENSUS_ROUNDS,
    claim_type: str = "state",
    temporal: str = "static",
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
        dry_run=dry_run, project=project, location=location,
        claim_type=claim_type, temporal=temporal, temperature=0.1
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
                dry_run=dry_run, project=project, location=location,
                claim_type=claim_type, temporal=temporal, temperature=0.7
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
