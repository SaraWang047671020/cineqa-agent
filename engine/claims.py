"""Claim Extraction: Converts complex director prompts & reference images into a 4-Tier Critical Path.
Enforces Saliency Budgeting (6-12 golden claims) to eliminate micro-rendering noise and token explosion.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from config.settings import settings

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_text": {"type": "string"},
                    "tier": {
                        "type": "string",
                        "enum": [
                            "tier1_causal_action",
                            "tier2_spatial_geometry",
                            "tier3_multimodal_consistency",
                            "tier4_physics_defect_control"
                        ],
                        "description": "The criticality tier of this visual constraint."
                    },
                    "type": {
                        "type": "string",
                        "enum": ["count", "direction", "relative_position",
                                  "relative_size", "color", "state", "action"],
                    },
                    "verifiable": {"type": "boolean"},
                    "temporal": {"type": "string", "enum": ["static", "sequential"]},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "reference_source": {
                        "type": "string", 
                        "enum": ["text_prompt", "storyboard_image", "character_concept_art", "cross_modal_joint"]
                    },
                    "importance_rationale": {
                        "type": "string",
                        "description": "Why this specific claim is critical to shot success."
                    }
                },
                "required": ["claim_text", "tier", "type", "verifiable", "temporal", "entities", "importance_rationale"],
            },
        }
    },
    "required": ["claims"],
}

TIERED_EXTRACTION_PROMPT = """You are a Lead AI Cinema VFX Supervisor and Script Supervisor.
Your task: Distill the director's prompt (which may contain hundreds of words of camera specs, style guidelines, and detailed narrative) into a high-precision, **4-Tier Critical Path Verification Schema**.

### Strict Rules for Cinematic Saliency & Budget Control:
1. **STRICT GOLDEN BUDGET**: Extract ONLY **6 to 12 high-impact, decisive claims**. Quality, criticality, and actionable verifiability over raw quantity.
2. **FILTER OUT UNCHECKABLE RENDER NOISE**:
   - DO NOT extract abstract style meta-tags (e.g. "8K IMAX", "photorealistic", "no 3D render", "180° shutter", "audio environmental SFX", "pore-level skin", "asymmetric moles").
   - DO NOT extract camera lens model names unless it describes an observable physical shot framing.
3. **CONSOLIDATE FRAGMENTED ATTRIBUTES**:
   - Merge scattered attributes into single coherent entity claims (e.g., "The projectile is a thin cream-grey bone shard dart with a sharp needle point" instead of 4 separate fragments).
4. **4-TIER CRITICAL PATH HIERARCHY**:
   - **`tier1_causal_action`**: Core chronological forward timeline (projectile entrance/exit, trigger event, instant physical reaction, wound opening, character clenching).
   - **`tier2_spatial_geometry`**: Strict spatial alignments and framing bounds (e.g., cut passes directly through navel center, face out of frame, left/right bokeh framing).
   - **`tier3_multimodal_consistency`**: Visual asset alignment with attached concept art / storyboard references (if provided).
   - **`tier4_physics_defect_control`**: Critical physical laws and negative defect suppressions (e.g., continuous non-stop flight, no bone embedded in skin, 24fps normal speed with no slow-mo).

Scene Prompt & Technical Directives:
{scene_text}

Output JSON matching the schema."""

def extract_claims(
    scene_text: str, 
    reference_image_paths: Optional[Union[str, List[str]]] = None, 
    reference_image_path: Optional[str] = None,
    project: Optional[str] = None, 
    location: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Calls Gemini via Vertex AI to extract a 4-Tier Critical Path of 6-12 decisive claims."""
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
    attached_count = 0
    for idx, img_path in enumerate(image_list):
        if img_path and os.path.exists(img_path):
            ext = Path(img_path).suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            with open(img_path, "rb") as fh:
                contents.append(
                    types.Part.from_bytes(data=fh.read(), mime_type=mime)
                )
            contents.append(f"[REFERENCE_IMAGE_{idx+1}: {Path(img_path).name}]")
            attached_count += 1

    if attached_count > 0:
        instruction = (
            TIERED_EXTRACTION_PROMPT.format(scene_text=scene_text) + 
            f"\n\n[NOTE]: {attached_count} official visual reference images are attached above. Include Tier 3 claims verifying cross-modal identity and storyboard composition against these images."
        )
        contents.append(instruction)
    else:
        contents.append(TIERED_EXTRACTION_PROMPT.format(scene_text=scene_text))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CLAIM_SCHEMA,
            temperature=0.1
        ),
    )
    return json.loads(response.text)["claims"]
