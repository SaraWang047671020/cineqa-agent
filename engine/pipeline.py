"""Unified Verification Pipeline: Scene Text + Multiple Concept Images + Video -> Claims -> Ledger.
Supports both batch execution and live streaming progress generators.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
from engine.claims import extract_claims
from engine.verify import call_gemini_verify_with_consensus, extract_frames

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
            return entry

        frame_out_dir = Path(frames_dir) / claim_id
        sampling = claim.get("sampling_strategy", "uniform")
        
        frames = extract_frames(video_path, str(frame_out_dir), claim["temporal"], claim_id, sampling_strategy=sampling, claim_type=claim.get("type", "action"))
        if not frames:
            entry["verdict"] = "CANNOT_DETERMINE"
            entry["observed"] = "Failed to sample frames from video take."
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
        entry["frame_obs"] = verdict_data.get("frame_observations", "")
        entry["causality"] = verdict_data.get("event_causal_order", "")
        entry["physics_laws"] = verdict_data.get("physics_law_grounding_check", "")
        entry["defect_frame_indices"] = verdict_data.get("defect_frame_indices", [])

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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
