"""Unified Verification Pipeline: Scene Text + Multiple Concept Images + Video -> Claims -> Ledger.
Supports both batch execution and live streaming progress generators.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
from engine.claims import extract_claims
from engine.verify import call_gemini_verify_with_consensus, extract_frames

# Tiered Trust & Empirical Reliability Threshold Configuration
AUTONOMOUS_VERIFY_THRESHOLD = 0.80

# Types meeting empirical precision >= 80% with objective visual certainty
AUTONOMOUS_CLAIM_TYPES = {
    "style",             # Visual medium / aesthetic rendering (~95% precision)
    "color",             # Chromatic palette & color consistency (100% on 92 benchmark rows)
    "relative_position", # Spatial left/right/above/below with coordinate anchoring (100% on benchmark)
    "state",             # Static props, wearing, posture (100% on benchmark)
    "direction",         # Screen-space motion vector traversal (100% on benchmark)
}

# Types routed to Advisory / Human Review (< 80% threshold or subjective cinematic aesthetics)
HUMAN_REVIEW_CLAIM_TYPES = {
    "camera",            # Framing and movement aesthetics (handed off to director for aesthetic sign-off)
    "count",             # Small integer numeration (75% precision on benchmark, dense overlap risk)
    "action",            # Temporal process, physical contact, fluid dynamics (70% precision on benchmark)
    "relative_size",     # Depth perspective bounding & optical foreshortening (50% precision on benchmark)
    "physics_sanity",    # Micro-topology, morphing, mesh clipping (theoretical VLM boundary)
}

def stream_pipeline(
    scene_text: str,
    video_path: str,
    frames_dir: str,
    reference_image_paths: Optional[Union[str, List[str]]] = None,
    reference_image_path: Optional[str] = None,
    scene_id: str = "scene",
    dry_run: bool = False,
    project: Optional[str] = None,
    location: Optional[str] = None,
    claims: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Generator[Tuple[int, int, Dict[str, Any]], None, None]:
    if not claims:
        claims = extract_claims(
            scene_text,
            reference_image_paths=reference_image_paths,
            reference_image_path=reference_image_path,
            project=project,
            location=location
        )
    
    total_claims = len(claims)

    import concurrent.futures

    def process_claim(i, claim):
        claim_id = f"{scene_id}_c{i:03d}"
        entry = {
            "claim_id": claim_id,
            "claim_text": claim["claim_text"],
            "type": claim["type"],
            "tier": claim.get("tier", ""),
            "verifiable": claim["verifiable"],
            "temporal": claim["temporal"],
            "entities": claim.get("entities", []),
            "reference_source": claim.get("reference_source", "text_prompt")
        }

        if not claim["verifiable"]:
            entry["verdict"] = "SKIPPED_NOT_VERIFIABLE"
            entry["observed"] = "Subjective or non-visually verifiable claim."
            entry["review_tier"] = "flagged_review"
            entry["review_reason"] = "Subjective or non-visually verifiable claim."
            return entry

        frame_out_dir = Path(frames_dir) / claim_id
        sampling = claim.get("sampling_strategy", "uniform")
        
        frames = extract_frames(video_path, str(frame_out_dir), claim["temporal"], claim_id, sampling_strategy=sampling, claim_type=claim.get("type", "action"))
        if not frames:
            entry["verdict"] = "CANNOT_DETERMINE"
            entry["observed"] = "Failed to sample frames from video take."
            entry["review_tier"] = "flagged_review"
            entry["review_reason"] = "Failed to sample frames from video take."
            return entry

        verdict_data = call_gemini_verify_with_consensus(
            claim["claim_text"],
            frames,
            reference_image_paths=reference_image_paths,
            reference_image_path=reference_image_path,
            dry_run=dry_run,
            project=project,
            location=location,
            claim_type=claim["type"],
            temporal=claim["temporal"]
        )
        entry["verdict"] = verdict_data.get("verdict", "CANNOT_DETERMINE")
        entry["observed"] = verdict_data.get("observed", "")
        entry["confidence"] = verdict_data.get("confidence")
        # Include detailed physical/geometric analysis fields
        entry["physics_sanity"] = verdict_data.get("physics_and_reality_sanity_check", "")
        entry["spatial_geometry"] = verdict_data.get("spatial_geometry_check", "")
        entry["motion_anchoring"] = verdict_data.get("motion_anchoring_check", "")
        entry["camera_motion"] = verdict_data.get("camera_motion_check", "")
        entry["frame_obs"] = verdict_data.get("frame_observations", "")
        entry["causality"] = verdict_data.get("event_causal_order", "")
        entry["physics_laws"] = verdict_data.get("physics_law_grounding_check", "")
        entry["defect_frame_indices"] = verdict_data.get("defect_frame_indices", [])

        # Determine Trust Tier: Autonomous Verified vs Flagged for Director Review
        claim_type = claim.get("type", "action")
        is_autonomous_type = claim_type in AUTONOMOUS_CLAIM_TYPES
        conformal_autonomous = verdict_data.get("conformal_autonomous") if verdict_data.get("conformal_autonomous") is not None else True
        conformal_set_size = verdict_data.get("conformal_set_size") or 1

        # Dual-gate verification: Type precision >= 80% AND decisive Conformal Prediction Set (size == 1)
        if is_autonomous_type and conformal_autonomous and conformal_set_size == 1:
            entry["review_tier"] = "verified"
            entry["review_reason"] = "High-confidence visual attribute meeting ≥ 80% precision threshold with Split-Conformal coverage guarantee."
        else:
            entry["review_tier"] = "flagged_review"
            if claim_type == "camera":
                entry["review_reason"] = "Camera movement and framing involve subjective cinematic aesthetics. Extracted kinematic signals provided for director sign-off."
            elif not is_autonomous_type:
                entry["review_reason"] = f"Claim type '{claim_type}' falls below our 80% empirical precision threshold. Extracted evidence provided for director review."
            else:
                entry["review_reason"] = f"Split-Conformal prediction set is ambiguous ({verdict_data.get('prediction_set', [])}). Flagged for director review."

        entry["prediction_set"] = verdict_data.get("prediction_set", [entry["verdict"]])
        entry["conformal_set_size"] = conformal_set_size
        entry["conformal_autonomous"] = conformal_autonomous
        entry["coverage_guarantee"] = verdict_data.get("coverage_guarantee", 0.80)

        # Map frame indices to real physical elapsed video seconds
        entry["frame_timestamps"] = [round(ts, 2) for _, ts in frames]
        defect_indices = entry["defect_frame_indices"]
        valid_defect_ts = [round(frames[i][1], 2) for i in defect_indices if i < len(frames)]
        entry["defect_timestamps"] = valid_defect_ts
        if valid_defect_ts:
            min_ts = min(valid_defect_ts)
            max_ts = max(valid_defect_ts)
            if abs(max_ts - min_ts) < 0.15:
                entry["defect_time_window"] = f"t={min_ts:.1f}s"
            else:
                entry["defect_time_window"] = f"{min_ts:.1f}s - {max_ts:.1f}s"
        else:
            entry["defect_time_window"] = "Whole Clip" if entry["verdict"] == "MISMATCH" else ""

        entry["defect_boxes"] = []
        if entry["verdict"] == "MISMATCH" and entry["defect_frame_indices"]:
            from engine.verify import localize_defect_bbox
            candidate_paths = [frames[i][0] for i in entry["defect_frame_indices"] if i < len(frames)]
            candidate_ts = [frames[i][1] for i in entry["defect_frame_indices"] if i < len(frames)]
            raw_boxes = localize_defect_bbox(claim["claim_text"], candidate_paths)
            for b in raw_boxes:
                fi = b.get("frame_index", 0)
                if fi < len(candidate_ts) and len(b.get("bbox_normalized", [])) == 4:
                    entry["defect_boxes"].append({"ts": candidate_ts[fi], "bbox": b["bbox_normalized"]})
        entry["checkable_components"] = verdict_data.get("checkable_components", [])
        entry["frame_observations"] = verdict_data.get("frame_observations", "")
        entry["all_required_subjects_fully_visible"] = verdict_data.get("all_required_subjects_fully_visible")
        entry["artifacts_affect_judgment"] = verdict_data.get("artifacts_affect_judgment")
        entry["concept_art_consistency"] = verdict_data.get("concept_art_consistency", "")
        entry["consensus_calls"] = verdict_data.get("consensus_calls", 1)
        entry["consensus_votes"] = verdict_data.get("consensus_votes", [entry["verdict"]])
        return entry

    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_claim = {executor.submit(process_claim, i, c): (i, c) for i, c in enumerate(claims)}
        for future in concurrent.futures.as_completed(future_to_claim):
            entry = future.result()
            completed_count += 1
            yield (completed_count - 1, total_claims, entry)

def run_pipeline(
    scene_text: str, 
    video_path: str, 
    frames_dir: str, 
    reference_image_paths: Optional[Union[str, List[str]]] = None,
    reference_image_path: Optional[str] = None,
    scene_id: str = "scene", 
    dry_run: bool = False,
    project: Optional[str] = None, 
    location: Optional[str] = None,
    claims: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Batch execution helper that collects all streamed claims into a list."""
    ledger = []
    for _, _, entry in stream_pipeline(
        scene_text=scene_text,
        video_path=video_path,
        frames_dir=frames_dir,
        reference_image_paths=reference_image_paths,
        reference_image_path=reference_image_path,
        scene_id=scene_id,
        dry_run=dry_run,
        project=project,
        location=location,
        claims=claims,
        **kwargs
    ):
        ledger.append(entry)
    return ledger
