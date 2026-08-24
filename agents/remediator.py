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

3. **Semantic Gradient Diagnosis & Targeted Inpainting** (arXiv:2603.12310):
   - Identify the exact failing token spans and defect time intervals (e.g. 00:01-00:03).
   - Recommend parameter tuning (CFG Scale, Motion Bucket ID) and provide actionable director advice.

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
    "estimated_dollars_saved": float,
    "remediation_summary": "string"
}
"""

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
                    temperature=0.2
                )
            )

            result = json.loads(response.text)
            elapsed = time.time() - start_time
            REMEDIATION_DURATION_SECONDS.observe(elapsed)

            # Record estimated savings in Prometheus
            saved = float(result.get("estimated_dollars_saved", 0.35))
            DOLLARS_SAVED_ESTIMATE.inc(saved)

            return result
