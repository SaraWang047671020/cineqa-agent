"""Frame Verification Engine: Evaluates atomic claims against video frames & reference concept art.
Implements 5-Step Mandatory Verification Protocol with high-density adaptive sampling,
continuous shot / no-cut disambiguation, fluid dynamics verification, and Split-Conformal (LAC) Conformal Decision Layer.
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2
from config.settings import settings
from agents.conformal_judge import ConformalJudge

VALID_VERDICTS = {"MATCH", "MISMATCH", "CANNOT_DETERMINE"}

PROMPT_BASE = """Claim to verify: "{claim_text}"\n\nThe video frames are attached in chronological order, each preceded by a label [FRAME_i @ t=X.XXs] giving its exact timestamp in the take. Use these timestamps to reason about sampling gaps, motion speed, and temporal continuity. (If reference storyboard/character concept art images are provided, they are labeled as REFERENCE_IMAGE_X).\n\nYou MUST execute the following 5-Step Mandatory Verification Protocol in exact sequential order before determining your final verdict:\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSTEP 1: SUB-FACT DECOMPOSITION & PHYSICAL/CINEMATIC DYNAMICS PRE-REGISTRATION\n• Break down the claim into atomic sub-facts that must each individually hold true.\n• Wearing/Holding relations: Check for physical contact/attachment. Verify there is no mesh clipping (e.g. blade clipping through palm instead of being gripped).\nSTEP 2: OBJECTIVE OBSERVATION & COORDINATE ANCHORING (NO PREMATURE CONCLUSIONS)\n• Screen-Space Orientation: Left and Right are ALWAYS defined strictly from the VIEWER'S PERSPECTIVE looking at the screen (Viewer Left / Viewer Right), NEVER the subject's anatomical perspective.\n• Horizontal Movement Anchoring: For motion claims, explicitly record the subject's horizontal position in the first frame vs final frame (e.g. Left Edge X:15% -> Right Edge X:85%). If the claim implies screen-space traversal (e.g., "left to right", "up to down", or any directional word), the subject MUST physically cross the screen\'s axis.\n• Camera Tracking & Background Reference (CRITICAL): If a subject remains centrally framed across multiple frames, you MUST observe the BACKGROUND. If the background is shifting continuously, the camera is tracking the subject, which proves world-space movement. HOWEVER, if the claim contains directional requirements (like "left to right" or "top to bottom"), a tracking shot where the subject stays in the center is a MISMATCH because they failed the screen-space traversal requirement!\n• Micro-Dynamics Threshold (CRITICAL): If verifying subtle movements (e.g., gentle breeze, breathing, fluttering fabric, rustling leaves), DO NOT expect large spatial displacement. Carefully inspect the contours, edges, and textures of the subjects across consecutive frames. Visible structural shifts, shape deformations, or texture blurring between frames MUST be recognized as valid movement (MATCH). Note that images are heavily compressed/downscaled for efficiency, so do not expect pixel-perfect preservation of micro-details. Do not falsely mark subtle but visible structural changes as a completely 'static scene'.\n\nSTEP 3: EVIDENCE SUFFICIENCY, INEQUALITY BOUNDING & INCIDENTAL ACTORS\n• Partial Visibility: A full 100% silhouette outline is not required if the specific relation/attribute is unmistakable.\n• Incidental Actor Tolerance: If the target entity and its verified action hold true in the primary action window, late-frame peripheral appearances (e.g. incidental hand entering frame, background ambient motion) do NOT invalidate the claim unless the claim explicitly specified 'in complete isolation' or 'no hands'.\n• Lower-Bound Claims ('at least N times bigger/taller'): If the cropped/partially visible entity is the one required to be larger and its visible portion already exceeds the threshold, unobserved portions only increase the true size -> counts as SUFFICIENT evidence.\n• Depth Perspective Check: Ensure entities being compared reside on the same focal/depth plane to rule out foreshortening/perspective distortions.\n\nSTEP 4: LOCALIZED AI ARTIFACT DISCRIMINATION\n• Set `artifacts_affect_judgment=true` ONLY if AI generation defects (morphing, limb fusion, blur) occur directly in the spatial-temporal ROI needed to verify this claim.\n• If the core sub-fact is clearly verifiable in clean frames/regions, unrelated background artifacts do NOT invalidate the judgment.\n\nSTEP 5: VERDICT FORMULATION & REVERSE-DIFFUSION CAUSAL CHECK\n• Output MATCH if and only if ALL sub-facts hold true with verified physical motion, topological continuity, and forward causality.\n• Output MISMATCH if any sub-fact is contradicted (e.g. hard scene cut, static blood texture instead of active flowing liquid), OR if a causal event displays reverse motion/un-breaking (e.g. debris assembling back into an intact object instead of shattering).\n• Output CANNOT_DETERMINE only if evidence is genuinely insufficient or localized artifacts obstruct evaluation. Do not abuse abstention out of generic caution.\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""



BLOCK_DIRECTION = """??Motion Direction & Camera Tracking (CRITICAL):
For directional claims (e.g. 'runs left to right', 'moves top to bottom'), you MUST explicitly compute the screen-space coordinates. 
If the subject remains centrally framed while the background shifts, they are NOT moving across the screen space, they are just moving in world space! This is a MISMATCH for directional screen-space claims.
You must output your numerical tracking data in the motion_anchoring_check field."""

BLOCK_ACTION = """??Implicit Entity Deduction (CRITICAL): You MUST infer unstated physical subjects logically required by the verbs. For example, if the text says 'is kicked', a 'foot/leg' MUST be visibly present. An object cannot act upon itself. Note: the AGENT may be off-frame — this tolerance applies ONLY to the visibility of the actor, NEVER to whether the action itself was observed. An unobserved process is CANNOT_DETERMINE, not MATCH.
??Action Execution vs. Static Pose (SUPER CRITICAL): You must STRICTLY differentiate between an 'Action' occurring and a 'State' existing. If the claim says 'drawing a sword', you MUST observe the kinematic process of the sword starting in the scabbard and being pulled out across the frames. If the subject is merely holding an already-drawn sword from Frame 0 and running around with it, the action of 'drawing' DID NOT OCCUR and you MUST mark MISMATCH. Just because an object exists (e.g. a sword) does NOT mean the verb (e.g. drawing, dropping, throwing) happened.
??Specific Action/Motion Verification: You MUST confirm the kinetic movement corresponding to the exact verb. If across ALL sampled frames the subjects only stand still, hold a pose, or move without performing the specific requested verb, mark as MISMATCH.
??PHYSICAL INTERACTION (CRITICAL): If the claim describes an interaction (collide, slice, hit, touch), you MUST observe ACTUAL PHYSICAL CONTACT and CONSEQUENCE. Merely moving towards each other is a MISMATCH!
??Process Verbs Require a Visible Intermediate (CRITICAL): Verbs like 'draw', 'unsheathe', 'open', 'lift', 'break', 'melt', 'transform' describe a TRANSITION PROCESS, not just a change of state. To output MATCH you MUST identify at least ONE sampled frame showing the subject MID-TRANSITION (e.g. the blade partially out of the scabbard, the door partly ajar, the object partly fractured) and cite that frame number explicitly in `action_execution_check`. If you can only identify a 'before' state and an 'after' state with NO frame capturing the transition itself, you MUST output CANNOT_DETERMINE — the process may have occurred between samples, but you did not observe it, and inferring it from a before/after difference is NOT verification. Do NOT describe a frame as showing the action unless the intermediate state is actually visible in that frame."""

BLOCK_SEQUENTIAL = """??Single Continuous Shot / No-Cut Claims (CRITICAL): The attached frames are DISCRETE SAMPLES taken at intervals from a single video take. Frame-to-frame position jumps or rapid camera pans between sampled frames are normal sampling intervals, NOT video cuts! An actual 'CUT' only occurs if there is an abrupt, total discontinuity in environment/scene (e.g. instant teleportation from outdoor roof to indoor room, or instant camera perspective teleportation with completely disconnected geometry). If lighting, environment, and subject motion remain topologically continuous across the timeline, judge as a continuous shot (MATCH).\n??Fluid & Liquid Dynamics: Liquid flow, dripping, or bleeding requires ACTIVE, CONTINUOUS DOWNWARD DISPLACEMENT of liquid across sequential frames under gravity. Static blood stains, painted red streaks, or stationary droplets that remain frozen in position do NOT constitute 'flowing/dripping' -> mark as fluid physics failure.\n??Temporal changes: Explicitly compare early frames (t0) vs late frames (t_end); check intermediate frames for transient micro-events.\n⚠️ Causal Completeness & Chronology (CRITICAL): You MUST differentiate between 'State-Change/Impact' verbs (e.g., collide, slice, break, explode, melt) and 'Continuous Motion' verbs (e.g., run, walk, fly). For State-Change/Impact actions, the video MUST show the transition from the 'before' state. If Frame 0 (the very first frame) shows the action is ALREADY in the aftermath or contact state (e.g., cars already touching, knife already inside tomato), you MUST mark MISMATCH due to missing approach/cause. HOWEVER, for Continuous Motion actions (running, flying), starting mid-action in Frame 0 is completely valid and expected (MATCH). (Also, while you may tolerate missing intermediate steps when judging causal order, you must NEVER tolerate an unobserved process when verifying if an action occurred — an unobserved transition is CANNOT_DETERMINE).
?? Topological Persistence & Anti-Morphing (CRITICAL PHYSICS CHECK): Objects must maintain strict volume, shape, and physical structure across the timeline! They must NOT magically transform, melt, suddenly change axis/orientation (e.g. from vertical crack to horizontal halves without tumbling), or morph their topology midway through the video. If an object undergoes an unprompted, impossible physical metamorphosis or structural drift (morphing instead of rigid-body physics), you MUST mark MISMATCH for severe physics violation."""


BLOCK_PHYSICS_LAWS = """🚨 FUNDAMENTAL PHYSICS LAWS CHECK (Gravity, Momentum, Rigid-Body):
Beyond hallucination artifacts, you MUST separately verify these physical laws hold:
1. Gravity & Support: Unsupported objects MUST fall/accelerate downward over time; they must NOT hover, float, or remain suspended without a visible support structure or stated cause (e.g. magic, propulsion) in the claim.
2. Momentum & Energy Conservation: A collision's aftermath MUST be proportional to its cause. A small/light object striking a large/heavy object should NOT send the large object flying with disproportionate force. Action MUST have a plausible, size/mass-appropriate reaction.
3. Rigid-Body Collision Consistency: When two solid objects collide, the deformation/damage MUST be consistent with material properties (e.g. metal dents rather than liquifies; glass shatters rather than bends). Do not accept a collision where neither object shows ANY consequence unless the claim explicitly describes a bounce/deflection.
4. Material-Appropriate Motion: Heavy/rigid objects must move with weighted, decelerating motion; light/flexible objects (cloth, hair, small debris) may move more freely. If an object's motion speed/behavior contradicts its apparent material and mass, flag it.
State explicitly which of these 4 laws are relevant to this claim, and whether each one holds. You MUST reference specific frame numbers as evidence, not general impressions. Only mark a violation if it directly involves the entities/action THIS claim describes — do not fail this claim due to physics issues affecting unrelated objects elsewhere in the frame."""

BLOCK_PHYSICS_SANITY = """🚨 UNIVERSAL PHYSICS & REALITY CHECK (ABSOLUTE OVERRIDE — SCOPED TO THIS CLAIM'S SUBJECT ONLY):
Before verifying the specific claim, you MUST evaluate the baseline physical reality of the SPECIFIC ENTITIES/REGION this claim is about — NOT the entire video frame. Even if the video perfectly matches the text prompt, you MUST output MISMATCH ONLY IF the entity/subject THIS CLAIM DESCRIBES is directly affected by ANY of these hallucinations:
⚠️ SAMPLING GAP CAVEAT (READ FIRST): The attached frames are SPARSE, DISCRETE samples — consecutive frames may be 1-2 SECONDS apart, not consecutive video frames. An object that is present in one frame and absent in the next has NOT necessarily "vanished": in 1-2 seconds it can be kicked away, fall out of frame, be occluded by smoke/dust/debris, or move behind another object. Likewise an object that appears broken in one frame and intact in a later frame may simply be a different part of the same object, or the camera/object may have rotated. You may ONLY report an object-permanence, morphing, or topology violation if the timestamps of the two frames in question are LESS THAN 0.4 SECONDS APART, or if there is positive visual evidence of the impossible transition itself (not merely a before/after difference). If the gap is larger and you are inferring the violation from a before/after difference alone, you MUST treat it as insufficient evidence and NOT fail the claim.
1. Object Permanence (Popping/Vanishing): The claim's subject spontaneously pops into existence from thin air or vanishes into nothingness between frames.
2. Anatomical Failure (Disembodied Objects): Clothing/armor/props belonging to THIS claim's subject act on their own without a physical body attached.
3. Clipping/Phasing: THIS claim's subject passes completely through another solid object without causing damage.
4. Body Pose Discontinuity: THIS claim's subject's limbs/joints snap into anatomically impossible angles or swap positions instantaneously between frames.
CRITICAL: Minor, unrelated artifacts elsewhere in the frame (background glitches, a different character's minor pose oddity, distant unrelated objects) that do NOT affect this claim's specific subject MUST NOT cause a MISMATCH here — note them only as a general observation, not a verdict-changing failure. Only fail THIS claim if the hallucination directly involves what this claim is actually testing.
If you observe a hallucination that DOES directly affect this claim's subject, document it in `frame_observations` and output MISMATCH."""

BLOCK_COUNT = """??Count claims: Count only prominent, salient entity instances. Ignore distant background noise pixels, but count distinct intended scene elements."""

BLOCK_REFERENCE = """??Multimodal Reference Art: If REFERENCE_IMAGE is present, anchor character hair/attire/layout against the reference art."""

BLOCK_STYLE = """??Aesthetic & Medium Style (CRITICAL): This claim checks the artistic visual medium (e.g., 2D animation, 3D CGI, photorealism). Analyze the rendering technique, shading (flat vs volumetric), lines, and overall aesthetic. If the prompt explicitly asks for '2D anime' but the output looks like a 3D video game or real-life live-action, you MUST mark this as MISMATCH! Do not compromise on art style!"""

BLOCK_POSITIONING = """📌 Absolute & Relative Positioning (CRITICAL VLM FIX):
Vision-Language Models inherently suffer from 'Spatial Blindness' and 'Yes-Man Bias'. If a claim asks about spatial layout, do NOT just spot both objects and output MATCH. You MUST execute this rigid coordinate check:
1. Identify the absolute X/Y centroid coordinate of Object A on the screen.
2. Identify the absolute X/Y centroid coordinate of Object B on the screen.
3. TEMPORAL PERSISTENCE (CRITICAL): The spatial relationship MUST hold true across ALL sampled frames. If Object A starts on the right, but moves to the FRONT, BEHIND, or LEFT of Object B by the last frame, you MUST mark MISMATCH.
  4. MATHEMATICAL CHECK: 
   - If the claim says 'A is to the LEFT of B', A's X-coordinate MUST be mathematically LESS THAN B's X-coordinate.
   - If the claim says 'A is to the RIGHT of B', A's X-coordinate MUST be mathematically GREATER THAN B's X-coordinate.
   - If the claim says 'A is ABOVE B', A's Y-coordinate MUST be physically HIGHER UP (closer to the top edge) than B's.
4. If the mathematical check fails, you MUST output MISMATCH. Do not make excuses for the model's failure."""

BLOCK_SPATIAL = """? Spatial & Geometric Verification (Depth-Aware 3D Mode):
VLMs natively suffer from 'yes-man' bias. You must act as a strict geometrical judge.
1. Universal Dimension Specificity (CRITICAL): You MUST map the comparative keyword (adjectives, nouns, or verbs) to its exact geometrical axis:
   - 'taller' / 'higher' / 'height' -> Compare ONLY the Vertical Y-Axis (top-to-bottom bounding box).
   - 'wider' / 'longer' / 'thicker' / 'width' / 'length' -> Compare ONLY the Horizontal X-Axis or Z-Axis.
   - 'larger' / 'bigger' / 'massive' / 'size' / 'area' / 'volume' -> Compare the Total Pixel Area or Estimated 3D Volume.
   Never conflate these! If A is 5x wider than B, it has a larger Area. But if the claim asks about 'height', and A's Y-axis is shorter than B's, you MUST output MISMATCH. Do not let Area trick you into validating Height.
2. 2D vs 3D Estimation: First, estimate the specific 2D percentage for the requested axis (e.g., 'A: 20% width, B: 40% width'). Next, assess their DEPTH PLANE.
3. Perspective Calibration: If A is in the background, it naturally occupies fewer 2D pixels. Use environmental clues to deduce its TRUE 3D dimensions.
4. Verdict Rules: 
   - If they are on the SAME depth plane and A fails the specific dimensional test, output MISMATCH.
   - If A is in the background and its 2D size is smaller, but 3D context proves it meets the claim, output MATCH.
Be rigorous! Only measure the exact geometrical axis requested!"""

def build_verify_prompt(claim_text: str, claim_type: str, temporal: str, has_reference: bool, frame_count: int = 0, max_gap: float = 0.0) -> str:
    blocks = [PROMPT_BASE.format(claim_text=claim_text)]
    # Physics sanity is a dedicated Tier-0 claim — do not re-run it on every unrelated claim.
    if claim_type == "physics_sanity":
        blocks.append(BLOCK_PHYSICS_SANITY)
    if claim_type == "action":
        blocks.insert(1, BLOCK_ACTION)
    if claim_type == 'direction':
        blocks.insert(1, BLOCK_DIRECTION)
    if any(w in claim_text.lower() for w in ['left', 'right', 'above', 'below', 'top', 'bottom', 'positioned']):
        blocks.insert(1, BLOCK_POSITIONING)
    if claim_type == "style":
        blocks.insert(1, BLOCK_STYLE)
    
    # Always include spatial block just in case the extraction misclassified a spatial claim as "state"
    blocks.insert(1, BLOCK_SPATIAL)
    
    if temporal == "sequential":
        blocks.insert(1, BLOCK_SEQUENTIAL)
        blocks.insert(1, BLOCK_PHYSICS_LAWS)
    if claim_type == "count":
        blocks.insert(1, BLOCK_COUNT)
    if has_reference:
        blocks.insert(1, BLOCK_REFERENCE)
    if frame_count:
        blocks.append(
            f"📏 EVIDENCE DENSITY: You have {frame_count} sampled frames, with a maximum gap of "
            f"{max_gap:.2f}s between consecutive frames. Calibrate your certainty accordingly — "
            f"with fewer than 8 frames or gaps above 0.4s, you CANNOT reliably judge frame-to-frame "
            f"continuity, object permanence, or morphing. In that case, restrict your verdict strictly "
            f"to what the claim itself asserts."
        )
    return "\n".join(blocks)
VERIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "checkable_components": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The specific atomic sub-facts that must each individually hold true",
        },
        "physics_and_reality_sanity_check": {
            "type": "string",
            "description": "Check ONLY whether THIS CLAIM'S OWN subject/entities are affected by: object permanence failures, anatomical failures, phasing/clipping, or body pose discontinuity. Unrelated defects elsewhere in the video (other objects, other moments, background) MUST NOT be reported here — they are handled by the dedicated Tier-0 physics claim. If this claim's own subject is unaffected, state that plainly."
        },
        "physics_passed": {
            "type": "boolean",
            "description": "True if THIS CLAIM'S OWN subject is free of the hallucination classes above. Judge only this claim's subject — do NOT set this false because of unrelated defects elsewhere in the video."
        },
        "physics_law_grounding_check": {
            "type": "string",
            "description": "For each of the 4 physics laws in BLOCK_PHYSICS_LAWS (gravity/support, momentum/energy conservation, rigid-body collision consistency, material-appropriate motion) that is relevant to this claim: state whether it holds, and CITE the specific frame number(s) and observation from `frame_observations` as evidence. Do not restate conclusions without citing which frame proves it."
        },
        "entity_presence_check": {
            "type": "string",
            "description": "Step 1 (Objects & Implicit Entities): Identify all explicit nouns. THEN, logically deduce any unstated physical entities required by the verbs (e.g., 'kicked' requires a 'foot'). Confirm exactly which of these explicit and implicit entities are visibly present."
        },
        "action_execution_check": {
            "type": "string",
            "description": "Step 2 (Verbs): Identify all actions (e.g., kicked, broken, swinging). For the entities found in Step 1, verify if these specific actions actually occurred between them."
        },
        "spatial_geometry_check": {
            "type": "string",
            "description": "Step 3 (Space/Size): For ANY spatial or size claims, you MUST write down the estimated percentage of Screen Height and Screen Width for each entity (e.g. 'Dog height: 20%, Hydrant height: 40%'). You must then state if the math proves the claim. If it contradicts, output MISMATCH."
        },
        "motion_anchoring_check": {
            "type": "string",
            "description": "For motion/direction claims: state subject's screen-space X position at first vs last frame (e.g. 'Subject X: 45%->48%'), AND separately state whether background shifted (camera tracking) between frames. If subject's own X% barely changed while background shifted, verdict MUST be MISMATCH for 'left to right' claims."
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
        "defect_frame_indices": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "ONLY when verdict is MISMATCH and the defect is visible in specific identifiable frames (not a whole-clip style/color issue): list the FRAME_i index numbers where the defect is most clearly visible (e.g. if [FRAME_3 @ ...] shows the problem, include 3). Temporal identification (WHICH frame) is significantly more reliable than exact spatial coordinates — focus on getting frame numbers right, do not guess pixel positions here. If not applicable, output []."
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
        "checkable_components", "physics_and_reality_sanity_check", "physics_passed", "physics_law_grounding_check", "entity_presence_check", "action_execution_check",
        "frame_observations", "kinetic_motion_and_context", "evidence_sufficiency_check", "motion_anchoring_check",
        "all_required_subjects_fully_visible", "artifacts_affect_judgment",
        "event_causal_order", "defect_frame_indices", "verdict", "observed", "confidence",
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

def extract_frames(video_path: str, out_dir_str: str, temporal: str, claim_id: str, sampling_strategy: str = "uniform", claim_type: str = "action") -> List[Tuple[str, float]]:
    """Extracts frames from video in a SINGLE fast ffmpeg pass."""
    video = Path(video_path)
    
    if video.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        import shutil
        out_dir = Path(out_dir_str)
        if out_dir.exists(): shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{claim_id}_storyboard{video.suffix}"
        shutil.copy2(video, img_path)
        return [(str(img_path), 0.0)]
        
    out_dir = Path(out_dir_str)
    import shutil
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    duration = 4.0
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
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
        # Dynamic sampling density (Item 4)
        if claim_type in ["color", "state", "count", "style", "existence"]:
            max_frames = 5
        else:
            max_frames = 20
            
        if sampling_strategy == "fast_burst":
            burst_duration = min(duration, 1.5)
            n = min(max_frames, max(5, int(burst_duration * 12)))
            timestamps = [burst_duration * i / max(1, n - 1) for i in range(n)]
        else:
            n = max(2, min(max_frames, int(duration / 0.20)))
            start, end = duration * 0.02, max(duration - 0.01, duration * 0.02)
            timestamps = [start + (end - start) * i / max(1, n - 1) for i in range(n)]

    # Single FFmpeg pass (Item 3 & 6)
    frame_paths = []
    # Build a complex filter graph to select exact frames

    
    import concurrent.futures
    
    def extract_one(i, ts):
        out_path = out_dir / f"{claim_id}_f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ss", f"{ts:.3f}",
             "-frames:v", "1", "-q:v", "5", "-vf", "scale='min(800,iw)':-2", str(out_path)],
            capture_output=True,
        )
        if out_path.exists():
            return (str(out_path), ts)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_one, i, ts) for i, ts in enumerate(timestamps)]
        for f in futures:
            res = f.result()
            if res:
                frame_paths.append(res)
                
    frame_paths.sort(key=lambda x: x[1])
    return frame_paths

def _compute_frame_motion_scores(frame_image_paths: List[str]) -> List[Optional[float]]:
    """Computes mean-absolute-pixel-difference between consecutive frames (grayscale, downsampled)
    as an objective, non-hallucinatable motion magnitude signal for the VLM to reason from."""
    scores: List[Optional[float]] = [None]
    prev_gray = None
    for path in frame_image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                if prev_gray is not None:
                    scores.append(None)
                continue
            small = cv2.resize(img, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                diff = float(np.mean(np.abs(gray - prev_gray)))
                scores.append(diff)
            prev_gray = gray
        except Exception:
            if prev_gray is not None:
                scores.append(None)
    while len(scores) < len(frame_image_paths):
        scores.append(None)
    return scores[:len(frame_image_paths)]

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

    motion_scores = _compute_frame_motion_scores([p for p, ts in frame_paths])

    for i, (p, ts) in enumerate(frame_paths):
        if i == 0 or motion_scores[i] is None:
            label = f"[FRAME_{i} @ t={ts:.2f}s]"
        else:
            level = "LOW (check for stalled/static motion)" if motion_scores[i] < 3.0 else \
                    "HIGH (check for cut/teleport/discontinuity)" if motion_scores[i] > 25.0 else "normal"
            label = f"[FRAME_{i} @ t={ts:.2f}s | pixel-diff vs prev frame: {motion_scores[i]:.1f} ({level})]"
        contents.append(label)
        with open(p, "rb") as fh:
            contents.append(
                types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg")
            )

    gaps = [frame_paths[i+1][1] - frame_paths[i][1] for i in range(len(frame_paths)-1)]
    contents.append(build_verify_prompt(
        claim_text, claim_type, temporal, bool(image_list),
        frame_count=len(frame_paths),
        max_gap=max(gaps) if gaps else 0.0
    ))

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

    raw_verdict_before = result.get("verdict")

    # HARD OVERRIDE: only the dedicated Tier-0 physics claim may be failed by the global physics gate.
    # Other claims must be judged on their own subject matter, not on unrelated video-wide defects.
    if claim_type == "physics_sanity" and result.get("physics_passed") is False:
        result["verdict"] = "MISMATCH"

    if raw_verdict_before != result.get("verdict"):
        print(f"[Verify] OVERRIDE fired: {raw_verdict_before} -> {result['verdict']} "
              f"(claim_type={claim_type}, physics_passed={result.get('physics_passed')})")

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
    into the Split-Conformal (LAC) Conformal Decision Layer for guaranteed risk control.
    """
    import concurrent.futures

    if dry_run:
        first = call_gemini_verify(
            claim_text, frame_paths,
            reference_image_paths=reference_image_paths,
            reference_image_path=reference_image_path,
            dry_run=dry_run, project=project, location=location,
            claim_type=claim_type, temporal=temporal, temperature=0.1
        )
        result = dict(first)
        result["consensus_calls"] = 1
        result["consensus_votes"] = ["MATCH"]
        result["prediction_set"] = ["MATCH"]
        result["conformal_autonomous"] = True
        return result

    # Item 1: Parallelize the 3 consensus votes!
    def run_vote(idx):
        temp = 0.1 if idx == 0 else 0.3
        return call_gemini_verify(
            claim_text, frame_paths,
            reference_image_paths=reference_image_paths,
            reference_image_path=reference_image_path,
            dry_run=dry_run, project=project, location=location,
            claim_type=claim_type, temporal=temporal, temperature=temp
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=consensus_rounds) as executor:
        futures = [executor.submit(run_vote, i) for i in range(consensus_rounds)]
        votes = [f.result() for f in futures]

    verdict_strings = [v.get("verdict", "CANNOT_DETERMINE") for v in votes]
    conformal_result = _conformal_judge.evaluate_verdict(verdict_strings)
    
    calibrated_verdict = conformal_result["verdict"]
    representative = next((v for v in votes if v.get("verdict") == calibrated_verdict), None)
    if representative is None:
        representative = dict(votes[0])
        representative["observed"] = f"[CONFORMAL OVERRIDE: Final verdict forced to {calibrated_verdict} due to high ambiguity. Below is the dissenting vote's observation, which may contradict the verdict:] " + representative.get("observed", "")
        representative["frame_observations"] = f"[CONFORMAL OVERRIDE] {representative.get('frame_observations', '')}"
    else:
        representative = dict(representative)
    
    result = dict(representative)
    result["verdict"] = calibrated_verdict
    result["consensus_calls"] = len(votes)
    result["consensus_votes"] = verdict_strings
    result["prediction_set"] = conformal_result.get("prediction_set", [calibrated_verdict])
    result["conformal_set_size"] = conformal_result.get("set_size", 1)
    result["conformal_autonomous"] = conformal_result.get("is_autonomous", True)
    result["coverage_guarantee"] = conformal_result.get("coverage_guarantee", 0.80)
    
    if not conformal_result["is_autonomous"]:
        result["observed"] += f" [Split-Conformal (LAC) Risk Control: Ambiguous Prediction Set {result['prediction_set']}]"
        
    return result


def localize_defect_bbox(
    claim_text: str,
    frame_image_paths: List[str],
    project: Optional[str] = None,
    location: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Dedicated, single-purpose spatial grounding call — isolated from the main
    verification schema because mixing bbox output into a 15-field JSON call
    is documented to produce near-random boxes (VideoZeroBench: <1% joint accuracy)."""
    from google import genai
    from google.genai import types
    import json

    client = genai.Client(vertexai=True, project=project or settings.GOOGLE_CLOUD_PROJECT,
                           location=location or settings.GOOGLE_CLOUD_LOCATION)
    contents = []
    for i, p in enumerate(frame_image_paths):
        contents.append(f"[CANDIDATE_FRAME_{i}]")
        with open(p, "rb") as fh:
            contents.append(types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg"))
    contents.append(
        f'The claim "{claim_text}" was violated in these frames. For EACH frame, output a tight '
        f'bounding box [ymin, xmin, ymax, xmax] normalized 0-1000 around ONLY the specific object/region '
        f'causing the violation (not the whole frame). Be precise — a box covering most of the frame is wrong.'
    )
    schema = {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_index": {"type": "integer"},
                        "bbox_normalized": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["frame_index", "bbox_normalized"],
                },
            }
        },
        "required": ["boxes"],
    }
    try:
        response = client.models.generate_content(
            model=settings.DEFAULT_GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=0.1),
        )
        return json.loads(response.text).get("boxes", [])
    except Exception as e:
        print(f"[Verify] Defect localization call failed: {e}")
        return []
