import json
import time
from typing import Dict, Any
from google import genai
from google.genai import types
from telemetry.tracer import tracer
from telemetry.metrics import REMEDIATION_DURATION_SECONDS, DOLLARS_SAVED_ESTIMATE
from config.settings import settings
from agents.mcp_client_wrapper import search_clickhouse_memory

REMEDIATION_PROMPT_TEMPLATE = """You are an elite AI Cinema Prompt Engineer and Inpainting Strategist, adhering to state-of-the-art video generation research (RAPO CVPR2025, T2V-CompBench, Physics-Aware Counterfactual Reasoning, and VQQA Semantic Gradients).

Your task: Given a director's original scene prompt and the detailed Verification Ledger containing failed claims (MISMATCH / CANNOT_DETERMINE), synthesize a surgical, deterministic remediation strategy.

Follow these strict scientific guidelines:
1. **5-Part Disentangled Positive Prompt Formula** (to eliminate attribute binding leakage and spatial drift):
   - [Camera & Optics]: Lens focal length, camera motion type, viewer-perspective pan/tracking trajectory.
   - [Subject & Binding Attributes]: Explicit mapping of colors, materials, and persistent states to specific entities.
   - [Action & Causal Trajectory]: Chronological 3-stage forward flow [Initial State -> Deterministic Action -> Physical Reaction].
   - [Spatial Hierarchy & Numeracy]: Explicit left/right viewer positioning, depth layers (foreground/background), exact integer counts without incidental clutter.
   - [Lighting & Physics Anchor]: Real-world illumination, reflections, particle dynamics, and material physics.

1.5. **Quantified Spatial Anchoring (applies to ALL spatial/direction/position fixes)**:
   Whenever a fix touches subject position, movement direction, or relative placement, you MUST express it as explicit screen-space percentages and timestamps (e.g., "X: 15% -> 85%", "t=0s -> t=3s"), matching the same coordinate language the verification step uses. Do NOT rely on vague qualitative terms ("moves to the right", "on the left side") alone — always pair them with a concrete percentage/timestamp anchor.

1.6. **Physical Video Seconds vs. Sampled Frames (CRITICAL)**:
   Sampled frame numbers (e.g. Frame 1, Frame 5, Frame 15) ARE NOT SECONDS! In video inspection, ~5 frames are sampled per second (~0.20s per frame). For example, Frame 5 is at ~1.0s, Frame 10 is at ~2.0s, and Frame 15 is at ~3.0s in a 4-second video. NEVER write timestamps exceeding the clip duration (e.g., NEVER write 00:05 or 10s for a 4s video). All timing in the prompt must reflect real video elapsed seconds.

2. **Counterfactual Negative Prompt Synthesis** (arXiv:2509.24702):
   - Deliberately include CONCRETE, VISUALLY DESCRIBABLE failure tokens drawn directly from the ledger\'s `observed`/`frame_observations` text for each failed claim — not abstract narrative concepts. Prefer visual nouns/adjectives a video model can actually suppress (e.g., "static subject, centered character, camera tracking shot, morphing limbs, fused geometry, sudden teleportation, attribute bleeding, vanishing props") over abstract phrasing like "reversed causal direction" alone — if a causal/temporal violation must be expressed, pair it with its concrete visual symptom (e.g., "debris flying backward and reassembling" rather than just "reversed causal direction").

3. **SSPO (Stage-Specific Prompt Optimization) Routing Table**:
   You MUST classify each failed claim by its `type` in the ledger and apply the EXACT corresponding Prompt Modification Strategy below (Operationalizing RAPO++):
      - IF type in ("action", "state"):
       [Strategy: Micro-Geometrical Transformation & Shot Sequencing] Video models lack the conceptual "world model" of complex physical destruction. You MUST explicitly dictate the exact geometrical changes and assign them to specific seconds/shots. Use VFX terminology ("breakaway prop"). Format strictly as: "00:00-00:01 (Shot 1: The Setup): [Describe solid geometry and position]. 00:01-00:02 (Shot 2: The Fracture): [Describe the exact geometrical breaking, e.g., the solid rectangle violently splits into jagged flying polygons and splinters]. 00:02-00:04 (Shot 3: The Aftermath): [Describe final resting position, e.g., the largest jagged chunk flies backward and rests flat against the background wall]."
       
     - IF type in ("relative_position", "direction"):
       [Strategy: Absolute Coordinate Anchoring & Camera Vector Forcing] Video models struggle with relative spatial verbs like "running down" or "beside". You MUST explicitly constrain the CAMERA ANGLE (e.g., "high-angle shot looking down") and the SUBJECT'S FACING DIRECTION. Use strict viewer-centric coordinates ("viewer-left", "foreground-right").
       • Screen-Space Traversal Claims (CRITICAL): If the failed claim requires the subject to move "left to right" / "right to left" / cross the screen, you MUST:
         1. In [Camera & Optics], explicitly state: "static locked-off camera, fixed frame, camera does NOT pan, track, or follow the subject" — UNLESS the original prompt explicitly called for a tracking shot, in which case instead state the subject's motion must be visibly faster/larger than the camera's pan so net screen-space displacement still occurs.
         2. In [Action & Causal Trajectory], state an explicit start/end screen-space percentage and timing using timestamp-level shot blocking. Format strictly as: "00:00-00:01 (Shot 1: Entry): [subject enters at screen-left X:10%]. 00:01-00:03 (Shot 2: Traversal): [subject traverses to screen-right X:90%, independent of any camera motion]".
         3. Add to the Negative Prompt: "camera tracking shot, panning camera following subject, subject remains centered while background moves, static subject position".
       Add incorrect positions/camera angles to the Negative Prompt.

     - IF type == "relative_size":
       [Strategy: Extreme Semantic Scale Anchoring & Forced Perspective] Generative models cannot do math (e.g., "4 times taller"). You MUST use extreme cinematic scaling descriptors. If A must be much larger than B, describe A as "gargantuan, towering, colossal, filling the entire frame" and B as "microscopic, tiny, barely visible at the feet of A". Explicitly dictate the camera's forced perspective (e.g., "Extreme low angle worm's-eye view looking up at the colossal A"). To ensure massive objects are NOT cropped out, explicitly command: "extreme wide shot, ultra-wide angle lens, entire subject fully visible within frame". Add "equal size, normal proportions, cropped, out of frame, close-up" to the Negative Prompt.
       
     - IF type in ("color", "state", "count"):
       [Strategy: Explicit Attribute Binding] To prevent attribute leakage (color/texture bleeding), you MUST place adjectives immediately adjacent to their nouns. Simplify sentence structure. Explicitly suppress incorrect traits in the Negative Prompt (e.g., if a sword should be blue, add "red sword, green sword" to negative).

     - IF type == "style":
       [Strategy: Aesthetic Weight Forcing & Negative Filtering] Video models determine style heavily by the first few words and negative prompts. You MUST prepend the exact stylistic keywords (e.g., "Studio Ghibli style 2D anime", "Cel-shaded flat colors") at the absolute beginning of the positive prompt. You MUST add contradictory styles to the negative prompt (e.g., "photorealistic, live-action, 3D CGI, volumetric rendering").
     
   - IF tier == "tier1_causal_action" OR temporal == "sequential":
     [Strategy: Chronological State Forcing] Enforce rigid temporal flow using explicit state transitions: "00:00 (Initial): [State A]. 00:01 (Action): [State B]. 00:03 (Final Result): [State C with explicit positions]." Suppress "reversed causal direction" or "simultaneous actions" in the Negative Prompt.

5. **CRITICAL PRESERVATION RULE (ANTI-HALLUCINATION)**:
   DO NOT alter any geometries, directions, or entities that are not explicitly marked as MISMATCH in the ledger. If the original prompt specifies 'running downward', 'left to right', etc., you MUST preserve these exact spatial and directional intents. Never flip directions (e.g., down to up, left to right) unless explicitly told the original was wrong.

4. **Targeted Token Surgery**:
   Identify the exact failing token spans based on the above mapping. When populating the `rationale` in the `targeted_token_surgery` output, explicitly state which [Strategy] you applied.
   • CRITICAL CONSISTENCY RULE: Every `repaired_phrase` listed in `targeted_token_surgery` MUST appear verbatim (or as a direct substring) inside `refined_positive_prompt`. Do not paraphrase the fix differently in the final prompt than what you listed in the surgery table — the two must be consistent.

6. **Conciseness Constraint**: For each individual fix, use 1-2 short, concrete declarative sentences. Do not stack multiple flowery adjectives or cinematic jargon onto a single fix — precision matters more than vividness. The final `refined_positive_prompt` should read as a sequence of clear, literal instructions, not a poetic description.

Output valid JSON matching this schema:
{
    "structured_prompt_breakdown": {
        "part_1_subject": "string",
        "part_2_action": "string",
        "part_3_setting": "string",
        "part_4_camera_movement": "string",
        "part_5_lighting": "string",
        "part_6_style": "string"
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
                "part_1_subject": {"type": "string"},
                "part_2_action": {"type": "string"},
                "part_3_setting": {"type": "string"},
                "part_4_camera_movement": {"type": "string"},
                "part_5_lighting": {"type": "string"},
                "part_6_style": {"type": "string"},
            },
            "required": ["part_1_subject", "part_2_action", "part_3_setting",
                         "part_4_camera_movement", "part_5_lighting", "part_6_style"],
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

    def remediate(self, original_prompt: str, current_take_prompt: str, inspection_result: Dict[str, Any], shot_id: str = "shot_001") -> Dict[str, Any]:
        start_time = time.time()
        with tracer.start_as_current_span("PromptRemediator.remediate"):
            user_content = (
                "Original Director's Scene Intent (ANCHOR - DO NOT DEVIATE FAR FROM THIS):\n" + original_prompt + "\n\n"
                "Prompt Used for Current Failed Take:\n" + current_take_prompt + "\n\n"
                "Inspection Report & Verification Ledger:\n" + json.dumps(inspection_result, indent=2)
            )

            # Agentic Memory Retrieval (Phase 2.1)
            # Find failed claim types to help the model decide what to query
            ledger = inspection_result.get("ledger", [])
            failed_types = set([c.get("type", "unknown") for c in ledger if c.get("verdict") == "MISMATCH"])
            
            tool_instruction = "\n\nYou MUST use the 'search_clickhouse_memory' tool to search for historical fixes for the following failed claim types before generating your final plan: " + ", ".join(failed_types) if failed_types else ""
            user_content += tool_instruction

            # Use an automated chat session to handle tool calls natively
            chat = self.client.chats.create(
                model=settings.DEFAULT_GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    tools=[search_clickhouse_memory],
                    temperature=0.2
                )
            )
            
            # Send the prompt and let Gemini autonomous call MCP!
            print(f"[Remediator] Triggering Agentic MCP Tool Calling for {failed_types}...")
            chat.send_message(user_content + "\n\n" + REMEDIATION_PROMPT_TEMPLATE)
            
            # Now ask for the final JSON payload
            response = chat.send_message(
                "Based on the tool results and the instructions, output the final JSON remediation plan.",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=REMEDIATION_RESPONSE_SCHEMA,
                    temperature=0.2
                )
            )
            
            # We don't need the old generate_content call
            if False:
                response = self.client.models.generate_content(
                model=settings.DEFAULT_GEMINI_MODEL,
                contents=[user_content, REMEDIATION_PROMPT_TEMPLATE],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=REMEDIATION_RESPONSE_SCHEMA,
                    temperature=0.2
                )
            )

            try:
                raw_text = response.text.strip()
                if raw_text.startswith("```json"):
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                result = json.loads(raw_text)
            except Exception as e:
                print(f"[Remediator] Fallback JSON parsing failed: {e}")
                result = {"refined_positive_prompt": original_prompt, "negative_prompt": ""}

            elapsed = time.time() - start_time
            REMEDIATION_DURATION_SECONDS.observe(elapsed)

            # Deterministic savings: each failed claim caught pre-regeneration
            # avoids one blind Veo retake at known cost
            VEO_COST_PER_TAKE = 0.40  # USD, Veo 3.1 Fast 4s take; adjust to actual pricing
            ledger = inspection_result.get("ledger") or (inspection_result if isinstance(inspection_result, list) else [])
            failed_count = sum(
                1 for c in ledger
                if isinstance(c, dict) and c.get("verdict") in ("MISMATCH", "CANNOT_DETERMINE")
            )
            saved = VEO_COST_PER_TAKE * max(1, failed_count)
            result["estimated_dollars_saved"] = saved
            DOLLARS_SAVED_ESTIMATE.inc(saved)

            return result
