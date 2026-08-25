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
    **kwargs
) -> Generator[Tuple[int, int, Dict[str, Any]], None, None]:
    """
    Generator that verifies claims one-by-one and yields (current_index, total_claims, verified_entry)
    for real-time streaming progress in UI or logs.
    """
    claims = extract_claims(
        scene_text, 
        reference_image_paths=reference_image_paths, 
        reference_image_path=reference_image_path,
        project=project, 
        location=location
    )
    total_claims = len(claims)

    for i, claim in enumerate(claims):
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
            yield (i, total_claims, entry)
            continue

        frame_out_dir = Path(frames_dir) / claim_id
        sampling = claim.get("sampling_strategy", "uniform")
        frames = extract_frames(video_path, str(frame_out_dir), claim["temporal"], claim_id, sampling_strategy=sampling)
        if not frames:
            entry["verdict"] = "CANNOT_DETERMINE"
            entry["observed"] = "Failed to sample frames from video take."
            yield (i, total_claims, entry)
            continue

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
        entry["checkable_components"] = verdict_data.get("checkable_components", [])
        entry["frame_observations"] = verdict_data.get("frame_observations", "")
        entry["all_required_subjects_fully_visible"] = verdict_data.get("all_required_subjects_fully_visible")
        entry["artifacts_affect_judgment"] = verdict_data.get("artifacts_affect_judgment")
        entry["concept_art_consistency"] = verdict_data.get("concept_art_consistency", "")
        entry["consensus_calls"] = verdict_data.get("consensus_calls", 1)
        entry["consensus_votes"] = verdict_data.get("consensus_votes", [entry["verdict"]])
        
        yield (i, total_claims, entry)

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
    **kwargs
) -> List[Dict[str, Any]]:
    """Batch execution helper that collects all streamed claims into a list."""
    return [
        entry for _, _, entry in stream_pipeline(
            scene_text=scene_text,
            video_path=video_path,
            frames_dir=frames_dir,
            reference_image_paths=reference_image_paths,
            reference_image_path=reference_image_path,
            scene_id=scene_id,
            dry_run=dry_run,
            project=project,
            location=location,
            **kwargs
        )
    ]
