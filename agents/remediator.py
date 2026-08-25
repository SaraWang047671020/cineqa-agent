import json
import time
from typing import Dict, Any
from google import genai
from google.genai import types
from telemetry.tracer import tracer
from telemetry.metrics import REMEDIATION_DURATION_SECONDS, DOLLARS_SAVED_ESTIMATE
from config.settings import settings

REMEDIATION_PROMPT_TEMPLATE = """You are an elite AI Cinema Prompt Engineer and Inpainting Strategist, adhering to state-of-the-art video generation research (RAPO CVPR2025, T2V-CompBench, Physics-Aware Counterfactual Reasoning, and VQQA Semantic Gradients).

Your task: Given a director's original scene prompt and the detailed Verification Ledger containing failed claims (MISMATCH / CANNOT_DETERMINE), synthesize a surgical, deterministic remediation strategy.

Follow these strict scientific guidelines:
1. **5-Part Disentangled Positive Prompt Formula** (to eliminate attribute binding leakage and spatial drift):
   - [Camera & Optics]: Lens focal length, camera motion type, viewer-perspective pan/tracking trajectory.
   - [Subject & Binding Attributes]: Explicit mapping of colors, materials, and persistent states to specific entities.
   - [Action & Causal Trajectory]: Chronological 3-stage forward flow [Initial State -> Deterministic Action -> Physical Reaction].
   - [Spatial Hierarchy & Numeracy]: Explicit left/right viewer positioning, depth layers (foreground/background), exact integer counts without incidental clutter.
   - [Lighting & Physics Anchor]: Real-world illumination, reflections, particle dynamics, and material physics.

2. **Counterfactual Negative Prompt Synthesis** (arXiv:2509.24702):
   - Deliberately include domain-specific counterfactual physics violations that directly address the observed failures in the ledger (e.g., "reversed causal direction, backwards motion, morphing limbs, fused geometry, sudden teleportation, attribute bleeding, vanishing props").

3. **SSPO (Stage-Specific Prompt Optimization) Routing Table**:
   You MUST classify each failed claim by its `type` in the ledger and apply the EXACT corresponding Prompt Modification Strategy below (Operationalizing RAPO++):
   
   - IF type == "action":
     [Strategy: Micro-Geometrical Transformation & Shot Sequencing] Video models lack the conceptual "world model" of complex physical destruction. You MUST explicitly dictate the exact geometrical changes and assign them to specific seconds/shots. Use VFX terminology ("breakaway prop"). Format strictly as: "00:00-00:01 (Shot 1: The Setup): [Describe solid geometry and position]. 00:01-00:02 (Shot 2: The Fracture): [Describe the exact geometrical breaking, e.g., the solid rectangle violently splits into jagged flying polygons and splinters]. 00:02-00:04 (Shot 3: The Aftermath): [Describe final resting position, e.g., the largest jagged chunk flies backward and rests flat against the background wall]."
     
   - IF type in ("relative_position", "direction", "relative_size"):
     [Strategy: Absolute Coordinate Anchoring] Video models struggle with relative spatial relations. You MUST use strict viewer-centric coordinates ("viewer-left", "foreground-right", "background-center"). Remove ambiguous prepositions. Add incorrect positions to the Negative Prompt.
     
   - IF type in ("color", "state", "count"):
     [Strategy: Explicit Attribute Binding] To prevent attribute leakage (color/texture bleeding), you MUST place adjectives immediately adjacent to their nouns. Simplify sentence structure. Explicitly suppress incorrect traits in the Negative Prompt (e.g., if a sword should be blue, add "red sword, green sword" to negative).
     
   - IF tier == "tier1_causal_action" OR temporal == "sequential":
     [Strategy: Chronological State Forcing] Enforce rigid temporal flow using explicit state transitions: "00:00 (Initial): [State A]. 00:01 (Action): [State B]. 00:03 (Final Result): [State C with explicit positions]." Suppress "reversed causal direction" or "simultaneous actions" in the Negative Prompt.

4. **Targeted Token Surgery**:
   Identify the exact failing token spans based on the above mapping. When populating the `rationale` in the `targeted_token_surgery` output, explicitly state which [Strategy] you applied.

Output valid JSON matching this schema:
{
    "structured_prompt_breakdown": {
        "camera_and_optics": "string",
        "subject_and_attributes": "string",
        "action_and_trajectory": "string",
        "spatial_and_numeracy": "string",
        "lighting_and_physics": "string"
    },
    "refined_positive_prompt": "string",
    "negative_prompt": "string",
    "targeted_token_surgery": [
        {
            "original_phrase": "string",
            "repaired_phrase": "string",
            "failure_claim_type": "string",
            "rationale": "string"
        }
    ],
    "parameter_tuning": {
        "cfg_scale": float,
        "motion_bucket_id": int,
        "inpaint_denoising_strength": float,
        "action_note": "string"
    },
    "suggested_inpaint_range": "string",
    "remediation_summary": "string"
}
"""

REMEDIATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "structured_prompt_breakdown": {
            "type": "object",
            "properties": {
                "camera_and_optics": {"type": "string"},
                "subject_and_attributes": {"type": "string"},
                "action_and_trajectory": {"type": "string"},
                "spatial_and_numeracy": {"type": "string"},
                "lighting_and_physics": {"type": "string"},
            },
            "required": ["camera_and_optics", "subject_and_attributes",
                         "action_and_trajectory", "spatial_and_numeracy",
                         "lighting_and_physics"],
        },
        "refined_positive_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "targeted_token_surgery": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_phrase": {"type": "string"},
                    "repaired_phrase": {"type": "string"},
                    "failure_claim_type": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["original_phrase", "repaired_phrase",
                             "failure_claim_type", "rationale"],
            },
        },
        "parameter_tuning": {
            "type": "object",
            "properties": {
                "cfg_scale": {"type": "number"},
                "motion_bucket_id": {"type": "integer"},
                "inpaint_denoising_strength": {"type": "number"},
                "action_note": {"type": "string"},
            },
        },
        "suggested_inpaint_range": {"type": "string"},
        "remediation_summary": {"type": "string"},
    },
    "required": ["refined_positive_prompt", "negative_prompt",
                 "targeted_token_surgery", "remediation_summary"],
}

class PromptRemediatorAgent:
    """
    Synthesizes surgical prompt adjustments, physics-aware negative prompts,
    and inpaint recommendations based on VQQA and RAPO prompt architectures.
    """
    def __init__(self, client: genai.Client = None):
        self.client = client or settings.get_genai_client()

    def remediate(self, original_prompt: str, inspection_result: Dict[str, Any], shot_id: str = "shot_001") -> Dict[str, Any]:
        start_time = time.time()
        with tracer.start_as_current_span("PromptRemediator.remediate"):
            user_content = (
                "Original Director Prompt:\n" + original_prompt + "\n\n"
                "Inspection Report & Verification Ledger:\n" + json.dumps(inspection_result, indent=2)
            )

            response = self.client.models.generate_content(
                model=settings.DEFAULT_GEMINI_MODEL,
                contents=[user_content, REMEDIATION_PROMPT_TEMPLATE],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=REMEDIATION_RESPONSE_SCHEMA,
                    temperature=0.2
                )
            )

            result = json.loads(response.text)
            elapsed = time.time() - start_time
            REMEDIATION_DURATION_SECONDS.observe(elapsed)

                        # Deterministic savings: each failed claim caught pre-regeneration
            # avoids one blind Veo retake at known cost
            VEO_COST_PER_TAKE = 0.40  # USD, Veo 3.1 Fast 4s take; adjust to actual pricing
            ledger = inspection_result.get("ledger", inspection_result if isinstance(inspection_result, list) else [])
            failed_count = sum(
                1 for c in ledger
                if isinstance(c, dict) and c.get("verdict") in ("MISMATCH", "CANNOT_DETERMINE")
            )
            saved = VEO_COST_PER_TAKE * max(1, failed_count)
            result["estimated_dollars_saved"] = saved
            DOLLARS_SAVED_ESTIMATE.inc(saved)

            return result
