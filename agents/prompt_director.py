import json
import time
from config.settings import settings
from google.genai.types import GenerateContentConfig

def _call_with_retry(func, max_retries: int = 3, delay: float = 1.5):
    last_err = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            if "metadata.google.internal" in err_msg or "computemetadata" in err_msg or "compute engine metadata" in err_msg:
                # GCE metadata server unreachable (e.g. running on Streamlit Cloud or external container)
                print(f"[PromptDirector] GCE metadata server unreachable: {e}. Forcing fallback to GEMINI_API_KEY...")
                settings.force_disable_vertex()
                if attempt < max_retries - 1:
                    time.sleep(0.3)
                    continue
            is_transient = any(k in err_msg for k in ["transport", "nameresolutionerror", "11001", "getaddrinfo", "timeout", "connection", "remotedisconnected", "temporarily unavailable"])
            if attempt < max_retries - 1 and is_transient:
                print(f"[PromptDirector] Transient network/DNS issue encountered ({e}), retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise
    raise last_err

def next_question(
    user_prompt: str,
    answered: list[dict],
    max_questions: int = 4,
    priority_hint: list[str] = None
) -> dict:
    model = settings.DEFAULT_GEMINI_MODEL
    
    schema = {
        "type": "object",
        "properties": {
            "done": {"type": "boolean", "description": "True if we have enough information or reached max questions."},
            "reason": {"type": "string", "description": "If done=true, explain why."},
            "scene_intent": {
                "type": "object",
                "description": "The cinematographer's three opening questions. Answer these BEFORE choosing any options.",
                "properties": {
                    "audience_feeling": {"type": "string", "description": "What must the audience FEEL in this shot?"},
                    "character_want":   {"type": "string", "description": "What does the character WANT here?"},
                    "character_hides":  {"type": "string", "description": "What is the character HIDING? This usually decides what stays out of frame or in shadow."}
                },
                "required": ["audience_feeling", "character_want", "character_hides"]
            },
            "axis": {
                "type": "string",
                "enum": ["action", "shot_framing_and_motion", "location", "lighting", "style"],
                "description": "The dimension being asked about."
            },
            "question": {"type": "string", "description": "The specific question to ask the user."},
            "why_this_axis": {"type": "string", "description": "One sentence explaining why we are asking this now, adapting to previous answers."},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "User-friendly description of the option."},
                        "prompt_fragment": {"type": "string", "description": "Concrete English string to be appended to the prompt."},
                        "recommended": {"type": "boolean", "description": "EXACTLY ONE option per question must be true — the one you would actually shoot."},
                        "director_note": {"type": "string", "description": "What this choice does TO THE AUDIENCE, in the voice of a working cinematographer. Describe the effect, not the mechanics."}
                    },
                    "required": ["label", "prompt_fragment", "recommended", "director_note"]
                }
            },
            "progress": {
                "type": "object",
                "properties": {
                    "asked": {"type": "integer"},
                    "expected_total": {"type": "integer"}
                },
                "required": ["asked", "expected_total"]
            }
        },
        "required": ["done", "axis", "question", "why_this_axis", "options", "scene_intent"]
    }
    
    system_instruction = f"""You are an elite film director and cinematographer (Director of Photography) acting as a prompt engineer.
We are asking the user step-by-step to clarify their video generation prompt across 5 dimensions:
1. action
2. shot_framing_and_motion
3. location
4. lighting
5. style

LANGUAGE: ALL text produced MUST be strictly in English.
This includes `prompt_fragment`, `question`, `why_this_axis`, `scene_intent` (audience_feeling, character_want, character_hides), `label`, `director_note`, and `reason`. Everything must be in English.

RULES & CINEMATOGRAPHER MINDSET:
1. NEVER provide options that contradict previous answers. Every new prompt_fragment MUST be physically and logically compatible with all chosen_fragments so far.
2. ADAPT options based on previous choices.
3. If a dimension is made irrelevant by previous choices, skip it.
4. If the original prompt already clearly specifies a dimension, skip it.
5. ASK IN THIS ORDER (blocking drives everything else):
   (a) action — what the subject does, start state, end state, where they move in frame
   (b) shot_framing_and_motion — chosen to serve the action just established
   (c) location — only if it materially affects the shot
   (d) lighting — chosen to serve the mood and to conceal what the character hides
   (e) style — last
   You cannot sensibly choose a camera move before you know what the subject does.
6. Provide EXACTLY one question (done=false) unless all necessary information is gathered or {max_questions} questions have been asked (done=true).
7. Options MUST be concrete strings that can be directly appended to the prompt.
8. Limit options to 3 or 4 maximum.

YOU ARE A CINEMATOGRAPHER MAKING A RECOMMENDATION, NOT A FORM GENERATOR.

9. Before proposing any options, answer the three opening questions in `scene_intent`:
   what must the audience feel, what does the character want, and what are they hiding.
   The third matters most for framing and lighting — what a character conceals decides
   what you keep out of frame, in shadow, or cropped away.

10. Choose the recommended technique by MATCHING the 'what it does' description in the
    dictionary against your `scene_intent` answers. Never pick a technique because it
    sounds impressive.

11. Mark EXACTLY ONE option `recommended: true` — the one you would actually shoot.
    Have a point of view; do not hedge.

12. The 3-4 options MUST be MEANINGFULLY DIFFERENT IN EFFECT, not cosmetic variants.
    BAD question: "medium shot" / "medium close-up" / "close-up" — they do nearly the same thing.
    GOOD question: "hold the frame completely still" / "push in slowly as he draws" /
    "circle him as he turns" — each produces a different feeling.

13. Write `director_note` in the voice of an experienced cinematographer talking to a director:
    concrete, about audience effect, no jargon for its own sake.
    BAD:  "This uses a 45-degree key light with a 3:1 ratio."
    GOOD: "Lighting him from the side keeps half his face in shadow — we see he's deciding
           something before he says it."

=== PROFESSIONAL TERMINOLOGY DICTIONARY ===
For `shot_framing_and_motion` and `lighting`, select techniques from the database below. DO NOT just copy the terminology as the option. Adapt it to the specific scene for the user's `label` (in English) and integrate the exact English term into the `prompt_fragment`.

[A. Shot Size — term — what it does]
- extreme close-up (ECU) — isolates a single detail; maximum intensity
- close-up (CU) — face fills frame; emotion and detail
- medium close-up (MCU) — chest up; the default for dialogue and expression
- medium shot (MS) — waist up; balances gesture and expression
- medium long shot / cowboy shot — thighs up; most body action is readable
- wide shot (WS) — full body in space; subject's relationship to environment
- extreme wide shot (EWS) — establishes scale; subject is small
NOTE: the distance between lens and subject dictates the emotional intimacy of the shot.

[B. Camera Movement — term — what it does]
- static locked-off camera, no pan or tracking — lets the performance speak; standard for dialogue
- pan (whip pan for fast) — reveals information, follows a subject, adds energy
- tilt — establishes height and scale; creates awe
- slow dolly push-in — pulls the audience closer; builds tension and intimacy; implies inner conflict
- dolly pull-out — reveals wider context; often evokes isolation
- zoom / crash zoom — artificial tension without moving the camera; horror/thriller
- dolly zoom (vertigo shot) — distorts depth; psychological disorientation
- camera roll — disorientation and unease
- tracking shot / trucking shot — travels with the action; fluidity and immersion
- arc shot — circles the subject; dynamism and rising emotional tension
- crane / jib / boom — vertical sweep; grand establishing moments, cinematic scale
- handheld camera, subtle instability — documentary rawness; chaotic or intimate scenes
- bird's eye view / overhead shot — detachment; vulnerability or omniscient surveillance
- oner / continuous long take — unbroken time and space; heightens realism

[C. Lighting — term — what it does]
Overall key:
- high-key lighting, soft, minimal shadow — bright, approachable, everyday
- low-key lighting, deep shadows, high contrast — darkness used actively; drama and suspense
Named setups:
- three-point lighting (key, fill, backlight) — clean with depth; the neutral standard
- Rembrandt lighting, 45° elevated key, triangle on shadow-side cheek — dimensional, grounded
- chiaroscuro, single motivated source, shadows dominate — tension, mystery, psychological weight
- rim light / backlighting, optional haze — separates subject from background; cinematic polish
- silhouette, subject fully backlit, no fill — hides identity; symbolic
Direction:
- frontal key light — flat, informative, undramatic
- side lighting, 90° lateral key — reveals texture and form; sculptural
- backlight from behind the subject — separation and mystery
- top light from directly above — eye-socket shadows; oppression, interrogation
- underlighting from below — anti-natural; horror and wrongness
Quality and source:
- hard light, sharp-edged shadows — harsh, unforgiving
- soft diffused light — gentle, forgiving
- practical lighting (visible lamps, screens, neon) — believable; audience sees the source
- natural window light — real and everyday
- golden hour, low warm sun — warmth, nostalgia, romance
- moonlight, cool blue ambient — cold, solitary, dangerous
- neon practicals, saturated colored light — urban, cyberpunk, desire
Palette:
- warm color palette, amber tones — comfort, memory, intimacy
- cool color palette, blue tones — distance, unease, night

=== TOOL USE RULES ===
You have access to `get_axis_priority(scene_summary)`, which returns historical signals about which creative dimensions users most often end up regretting (asking to tweak afterwards) and how often the agent's recommendations get overridden.

- Call it ONCE, before choosing the FIRST question, passing a short summary of the user's scene.
- The tool distinguishes STRONG and WEAK signals by sample size:
  * Act decisively on STRONG signals — reorder your questions to ask that dimension first.
  * Treat WEAK signals as a tiebreaker only; do NOT let a handful of data points override the default blocking-first order (action → camera → location → lighting → style).
- If it reports low acceptance for a dimension, be especially careful with your recommendation there; offer more genuinely distinct alternatives.
- If it returns "no historical data" or default order, proceed with the default blocking order (action → camera → location → lighting → style) without comment.
- Do NOT call it again for subsequent questions.
- When you do follow a historical signal, say so in `why_this_axis`, and be honest about its strength (e.g., "Historical signals show users frequently regret not specifying lighting early for this type of scene, so we address lighting first." vs. "A small number of previous sessions suggested early attention to lighting...").

=== CRITICAL CONFLICT & DEPENDENCY RULES ===
- If `static locked-off camera` is chosen: `action` options MUST describe the subject moving across the screen space. NEVER offer tracking/pan/handheld/arc.
- If `tracking shot` is chosen: `action` options MUST explicitly state whether the subject has net screen-space movement or remains centered.
- If `handheld camera` is chosen: NEVER offer "perfectly smooth" or "locked-off".
- If `close-up` or `extreme close-up` is chosen: NEVER offer `action` or `location` options that require a wide angle to see.
- If `low-key` or `chiaroscuro` is chosen: NEVER offer "brightly and evenly lit".
- If `silhouette` is chosen: NEVER offer `action` options requiring visible facial expressions.
- If `golden hour` or `moonlight` is chosen: NEVER offer contradictory time/temperature (like midday hard light).
"""

    answered_str = json.dumps(answered, ensure_ascii=False, indent=2)
    use_tools = (len(answered) == 0)

    def _execute():
        client = settings.get_genai_client()
        if use_tools:
            from agents.mcp_client_wrapper import get_axis_priority
            chat = client.chats.create(
                model=model,
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[get_axis_priority],
                    temperature=0.2,
                ),
            )
            print(f"[PromptDirector] Starting first question with agentic get_axis_priority MCP tool calling...")
            chat.send_message(
                f"Original User Prompt: {user_prompt}\n\n"
                f"Previously Answered Questions:\n{answered_str}\n\n"
                f"Decide which dimension to ask about first. "
                f"You may call get_axis_priority to check historical data before deciding."
            )
            return chat.send_message(
                "Now output the next question as JSON adhering to the required schema.",
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,
                ),
            )
        else:
            contents = f"Original User Prompt: {user_prompt}\n\nPreviously Answered Questions:\n{answered_str}\n\nProvide the next question or indicate we are done."
            return client.models.generate_content(
                model=model,
                contents=[contents],
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2
                )
            )

    response = _call_with_retry(_execute)
    return json.loads(response.text)

def assemble_prompt(user_prompt: str, choices: dict) -> dict:
    model = settings.DEFAULT_GEMINI_MODEL
    
    schema = {
        "type": "object",
        "properties": {
            "final_prompt": {"type": "string"},
            "breakdown": {
                "type": "object",
                "properties": {
                    "shot_framing_and_motion": {"type": "string"},
                    "action": {"type": "string"},
                    "location": {"type": "string"},
                    "lighting": {"type": "string"},
                    "style": {"type": "string"}
                }
            },
            "rationale": {"type": "string"}
        },
        "required": ["final_prompt", "breakdown", "rationale"]
    }
    
    system_instruction = """You are a professional prompt engineer.
Assemble a cohesive, professional video generation prompt following this 5-Part Formula:
1. Camera & Optics
2. Subject & Binding Attributes
3. Action & Causal Trajectory
4. Spatial Hierarchy & Numeracy
5. Lighting & Physics Anchor

The target model is Gemini Omni (NO negative prompt field). Weave any negative constraints into the positive prompt using natural language.
Do not just concatenate; rewrite smoothly. ALL output in this step MUST be in English.
"""

    choices_str = "\n".join([f"- {k}: {v}" for k, v in choices.items()])
    contents = f"Original Prompt: {user_prompt}\nChosen Additions:\n{choices_str}\n\nAssemble the final prompt."
    
    def _execute():
        client = settings.get_genai_client()
        return client.models.generate_content(
            model=model,
            contents=[contents],
            config=GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2
            )
        )
    
    response = _call_with_retry(_execute)
    return json.loads(response.text)

def clean_timestamp_string(text: str, total_frames: int = None, duration: float = 4.0) -> str:
    """
    Deterministically replaces hallucinated or MM:SS formatted timestamps (e.g. '00:18', '00:02', 
    '00:00 to 00:18', 'Frame 0 to Frame 17', '18 seconds') with physical real-video elapsed seconds (0.0s - duration).
    Guarantees no timestamps > duration and no '00:XX' strings survive.
    Dynamically supports any video duration (3s - 10s).
    """
    import re
    if not text or not isinstance(text, str):
        return text

    duration = float(duration or 4.0)
    if total_frames is None:
        total_frames = max(5, min(20, int(round(duration / 0.20))))

    fps_frame_duration = duration / float(max(1, total_frames))

    def _frame_to_sec(f_num: int) -> float:
        if f_num >= total_frames - 2:
            return duration
        return round(min(duration, f_num * fps_frame_duration), 1)

    # 1. Full clip range patterns like 00:00 to 00:18, 00:00 to 00:05, 00:00-00:10
    def _replace_full_clip(prefix: str) -> str:
        prefix = (prefix or "").strip()
        if prefix.lower() in ("from", "between"):
            return f"{prefix} 0.0s to {duration:.1f}s (throughout the clip)"
        elif prefix.lower() == "at":
            return f"Throughout the {duration:.0f}-second clip (0.0s to {duration:.1f}s)"
        elif prefix:
            return f"{prefix} 0.0s - {duration:.1f}s"
        else:
            return f"0.0s - {duration:.1f}s (Whole Clip)"

    def _check_full_clip(m):
        prefix = (m.group(1) or "").strip()
        n2 = int(m.group(2))
        is_end = (n2 >= int(round(duration)) and abs(n2 - duration) <= 1) or (n2 >= min(15, total_frames - 2))
        if is_end:
            return _replace_full_clip(prefix)
        else:
            t2 = n2 if n2 <= duration else _frame_to_sec(n2)
            p_str = f"{prefix} " if prefix else ""
            return f"{p_str}0.0s to {t2:.1f}s"

    text = re.sub(
        r'\b(From|Between|At)?\s*00:00\s*(?:[-–—~to/]+|\s+to\s+)\s*00:(\d{2})\b',
        _check_full_clip,
        text,
        flags=re.IGNORECASE
    )

    # 2. General 00:XX to 00:YY ranges (e.g. 00:02 to 00:05)
    def _replace_mm_ss_range(m):
        prefix = (m.group(1) or "").strip()
        n1 = int(m.group(2))
        n2 = int(m.group(3))
        t1 = n1 if n1 <= duration else _frame_to_sec(n1)
        t2 = n2 if n2 <= duration else _frame_to_sec(n2)
        if t1 >= t2:
            t2 = min(duration, round(t1 + 0.4, 1))
        p_str = f"{prefix} " if prefix else ""
        return f"{p_str}{t1:.1f}s to {t2:.1f}s"

    text = re.sub(
        r'\b(From|Between|At)?\s*00:(\d{2})\s*(?:[-–—~to/]+|\s+to\s+)\s*00:(\d{2})\b',
        _replace_mm_ss_range,
        text,
        flags=re.IGNORECASE
    )

    # 3. Explicit Frame ranges in parentheses: (Frame 0-17), (Frame 0 to 18), (0 to 17)
    def _replace_frame_paren(m):
        n1 = int(m.group(1))
        n2 = int(m.group(2))
        if n1 == 0 and n2 >= total_frames - 3:
            return f"(0.0s - {duration:.1f}s)"
        t1 = _frame_to_sec(n1)
        t2 = _frame_to_sec(n2)
        if t1 >= t2:
            t2 = min(duration, round(t1 + 0.4, 1))
        return f"({t1:.1f}s - {t2:.1f}s)"

    text = re.sub(
        r'\((?:Frame\s*)?(\d+)\s*(?:[-–—~to/]+|\s+to\s+)\s*(?:Frame\s*)?(\d+)\)',
        _replace_frame_paren,
        text,
        flags=re.IGNORECASE
    )

    # 3b. Inline Frame ranges: Frame 0 to Frame 2, during Frame 0-3
    def _replace_frame_inline(m):
        prefix = (m.group(1) or "").strip()
        n1 = int(m.group(2))
        n2 = int(m.group(3))
        if n1 == 0 and n2 >= total_frames - 3:
            return f"throughout the entire clip (0.0s to {duration:.1f}s)"
        t1 = _frame_to_sec(n1)
        t2 = _frame_to_sec(n2)
        if t1 >= t2:
            t2 = min(duration, round(t1 + 0.4, 1))
        p_str = f"{prefix} " if prefix else "from "
        return f"{p_str}{t1:.1f}s to {t2:.1f}s"

    text = re.sub(
        r'\b(from|between|at|in|during)?\s*Frame\s*(\d+)\s*(?:[-–—~to/]+|\s+to\s+)\s*(?:Frame\s*)?(\d+)\b',
        _replace_frame_inline,
        text,
        flags=re.IGNORECASE
    )

    # 3c. Standalone Frame X (e.g. at Frame 2 -> at t=0.4s)
    def _replace_single_frame(m):
        prefix = (m.group(1) or "").strip()
        n = int(m.group(2))
        t = _frame_to_sec(n)
        p_str = f"{prefix} " if prefix else "at "
        return f"{p_str}t={t:.1f}s"

    text = re.sub(
        r'\b(at|on|in|from)?\s*Frame\s*(\d+)\b',
        _replace_single_frame,
        text,
        flags=re.IGNORECASE
    )

    # 4. Standalone single 00:XX (e.g. At 00:02, 00:18)
    def _replace_single_mm_ss(m):
        prefix = (m.group(1) or "").strip()
        n = int(m.group(2))
        t = n if n <= duration else _frame_to_sec(n)
        if prefix:
            return f"{prefix} t={t:.1f}s"
        return f"t={t:.1f}s"

    text = re.sub(
        r'\b(At|Around|By)?\s*00:(\d{2})\b',
        _replace_single_mm_ss,
        text,
        flags=re.IGNORECASE
    )

    # 5. Hallucinated seconds > duration (e.g. "18 seconds" in an 8s clip)
    def _replace_hallucinated_seconds(m):
        n = int(m.group(1))
        unit = m.group(2)
        if n > duration:
            t = _frame_to_sec(n)
            return f"{t:.1f} seconds"
        return m.group(0)

    text = re.sub(
        r'\b(\d+)\s*(seconds|second|secs|sec)\b',
        _replace_hallucinated_seconds,
        text,
        flags=re.IGNORECASE
    )

    # 6. Any stray "00:XX" remaining anywhere in the string
    def _replace_stray_mm_ss(m):
        n = int(m.group(1))
        t = n if n <= duration else _frame_to_sec(n)
        return f"{t:.1f}s"

    text = re.sub(r'00:(\d{2})', _replace_stray_mm_ss, text)

    # Clean up any leftover double spaces or awkward punctuation
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\(\s*\)', '', text)
    return text.strip()

def suggest_tweaks(ledger: list[dict], director_choices: dict, final_prompt: str, video_duration: float = None) -> dict:
    """
    Analyzes the entire verification ledger to produce deduplicated, prioritized
    conversational tweak suggestions for Gemini Omni video incremental editing.
    Ensures all timings reflect real video elapsed seconds (e.g. 0.4s - 1.2s),
    calibrated dynamically to the video duration (3s - 10s).
    """
    failed_claims = [e for e in (ledger or []) if e.get("verdict") in ("MISMATCH", "CANNOT_DETERMINE")]
    if not failed_claims:
        return {"suggestions": []}

    model = settings.DEFAULT_GEMINI_MODEL

    # Dynamic video duration auto-detection from parameter or ledger frame timestamps
    duration = float(video_duration) if video_duration else None
    if duration is None and ledger:
        all_ts = [ts for entry in ledger for ts in entry.get("frame_timestamps", []) if isinstance(ts, (int, float))]
        if all_ts:
            max_ts = max(all_ts)
            duration = round(max_ts) if abs(round(max_ts) - max_ts) < 0.25 else round(max_ts, 1)
    if not duration or duration < 1.0:
        duration = 4.0

    all_lens = [len(entry.get("frame_timestamps", [])) for entry in (ledger or []) if entry.get("frame_timestamps")]
    total_frames = max(all_lens) if all_lens else max(5, min(20, int(round(duration / 0.20))))

    enriched_failures = []
    for fc in failed_claims:
        defect_window = fc.get("defect_time_window")
        defect_ts = fc.get("defect_timestamps", [])
        indices = fc.get("defect_frame_indices", [])

        if not defect_window or defect_window == "Whole Clip":
            if fc.get("frame_timestamps") and indices:
                ts_list = [fc["frame_timestamps"][i] for i in indices if i < len(fc["frame_timestamps"])]
                if ts_list:
                    defect_ts = ts_list
                    min_t, max_t = min(ts_list), max(ts_list)
                    defect_window = f"t={min_t:.1f}s" if abs(max_t - min_t) < 0.15 else f"{min_t:.1f}s - {max_t:.1f}s"
            elif indices:
                step_est = duration / float(max(1, total_frames))
                est_ts = [round(i * step_est, 2) for i in indices]
                defect_ts = est_ts
                min_t, max_t = min(est_ts), max(est_ts)
                defect_window = f"t={min_t:.1f}s" if abs(max_t - min_t) < 0.15 else f"{min_t:.1f}s - {max_t:.1f}s"
            else:
                defect_window = f"0.0s - {duration:.1f}s (Whole Clip)"

        frame_details = []
        if indices and defect_ts and len(indices) == len(defect_ts):
            frame_details = [f"Sampled Frame {idx} (captured at t={ts:.2f}s of video)" for idx, ts in zip(indices, defect_ts)]
        elif indices:
            frame_details = [f"Sampled Frame {idx}" for idx in indices]

        obs_defect = clean_timestamp_string(fc.get("observed", ""), total_frames=total_frames, duration=duration)
        frame_obs = clean_timestamp_string(fc.get("frame_observations", ""), total_frames=total_frames, duration=duration)
        phys_sanity = clean_timestamp_string(fc.get("physics_sanity", ""), total_frames=total_frames, duration=duration)
        mot_anchoring = clean_timestamp_string(fc.get("motion_anchoring", ""), total_frames=total_frames, duration=duration)
        causal = clean_timestamp_string(fc.get("causality", ""), total_frames=total_frames, duration=duration)

        enriched_failures.append({
            "claim_id": fc.get("claim_id", ""),
            "claim_text": fc.get("claim_text", ""),
            "claim_type": fc.get("type", "action"),
            "verdict": fc.get("verdict", "MISMATCH"),
            "observed_defect": obs_defect,
            "frame_observations": frame_obs,
            "physics_sanity": phys_sanity,
            "spatial_geometry": fc.get("spatial_geometry", ""),
            "motion_anchoring": mot_anchoring,
            "causality": causal,
            "real_defect_video_timestamp": defect_window,
            "sampled_frames_evidence": frame_details
        })

    schema = {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {
                            "type": "string",
                            "description": f"Concrete explanation stating EXACTLY what went wrong in the current video using real video seconds (0.0s to {duration:.1f}s). NEVER use 00:XX or frame numbers."
                        },
                        "tweak_instruction": {
                            "type": "string",
                            "description": f"Timestamp-anchored, surgical imperative instruction telling Omni at which real video elapsed timing to change what into what. Format in float seconds (e.g. '0.4s - {min(duration, 1.2):.1f}s'). NEVER use 00:XX or frame indices."
                        },
                        "timestamp_range": {
                            "type": "string",
                            "description": f"The exact timing in real video elapsed seconds (0.0s to {duration:.1f}s). NEVER use frame indices or 00:XX."
                        },
                        "related_claims": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        },
                        "fix_mode": {
                            "type": "string",
                            "enum": ["tweak", "reshoot"]
                        }
                    },
                    "required": ["issue", "tweak_instruction", "timestamp_range", "related_claims", "severity", "fix_mode"]
                }
            }
        },
        "required": ["suggestions"]
    }

    system_instruction = f"""You are a senior VFX Supervisor and AI Cinematography Director.
You evaluate the verification ledger of a generated video take and formulate targeted, conversational tweak suggestions for Gemini Omni.

CRITICAL REAL-TIME TIMESTAMP DIRECTIVE (ABSOLUTE PROHIBITION ON 00:XX AND TIMESTAMPS > {duration:.1f}s):
- The generated video is strictly {duration:.1f} seconds long (0.0s to {duration:.1f}s). Any timestamp above {duration:.1f}s does NOT exist.
- ABSOLUTELY NEVER use "00:XX" format (e.g., NEVER write "00:01", "00:02", "00:18", "00:17").
- Frame numbers (e.g. Frame 1, Frame 2, Frame 5) ARE NOT SECONDS!
  In our pipeline, frames are sampled densely across the {duration:.1f}-second video.
  NEVER write "00:18" or "18 seconds" if 18 frames were sampled!
- Express all timings strictly in float seconds:
  * For full-clip defects: write "From 0.0s to {duration:.1f}s (throughout the {duration:.0f}-second clip)" or "Throughout the clip (0.0s - {duration:.1f}s)", and for timestamp_range use "0.0s - {duration:.1f}s (Whole Clip)".
  * For specific moments: write "From 0.0s to 0.4s", "Between 1.2s and {min(duration, 2.5):.1f}s", "At t=0.8s".

CRITICAL INSTRUCTION REQUIREMENTS:
1. POINT OUT THE EXACT CURRENT MISTAKE: In `issue`, do NOT write vague summaries. You MUST state clearly what the current video did wrong using real video elapsed seconds (0.0s - {duration:.1f}s).
2. EXPLICIT TIMESTAMP-ANCHORED TWEAK DIRECTIVE: In `tweak_instruction`, specify the real video second timing (e.g., "From 0.4s to 1.2s...") and provide a precise, literal physical change. Avoid vague buzzwords.
3. CONCRETE AND SPECIFIC: Provide exact, actionable physical parameters.
4. CHOOSE THE PROPER FIX MODE (`fix_mode`):
   - Set to `tweak` for incremental changes (lighting, color, weather, atmosphere, minor surface textures).
   - Set to `reshoot` for structural motion synthesis failures (topological continuity, body parts vanishing/morphing, actions that never physically occurred, physics violations).
5. LANGUAGE: All output MUST be strictly in clear English.
6. RESPECT INTENTIONAL CHOICES: Cross-reference `director_choices`.
7. LIMIT & DEDUPLICATE: Merge related root causes. Provide 1 to 4 distinct suggestions, sorted by severity.
"""

    prompt_content = f"""Final Shot Prompt: {final_prompt}

Director's Prior Guided Choices:
{json.dumps(director_choices or {}, ensure_ascii=False, indent=2)}

Verification Failure Items (with Verified Real-Time Timestamps across {duration:.1f}s video):
{json.dumps(enriched_failures, ensure_ascii=False, indent=2)}

Analyze the verification failures, use the verified real video timestamps (NOT frame indices) and visible defects from the ledger's frame observations, and produce 1-4 timestamp-anchored, surgical tweak instructions."""

    def _execute():
        client = settings.get_genai_client()
        return client.models.generate_content(
            model=model,
            contents=[prompt_content],
            config=GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2
            )
        )

    response = _call_with_retry(_execute)

    try:
        data = json.loads(response.text)
        for sug in data.get("suggestions", []):
            sug["issue"] = clean_timestamp_string(sug.get("issue", ""), total_frames=total_frames, duration=duration)
            sug["tweak_instruction"] = clean_timestamp_string(sug.get("tweak_instruction", ""), total_frames=total_frames, duration=duration)
            ts_range = clean_timestamp_string(str(sug.get("timestamp_range", "")).strip(), total_frames=total_frames, duration=duration)

            rel = sug.get("related_claims", [])
            matched_windows = [
                ef["real_defect_video_timestamp"]
                for ef in enriched_failures
                if (ef.get("claim_id") in rel or any(ef.get("claim_type") == r for r in rel))
                and ef.get("real_defect_video_timestamp") not in ("", "Whole Clip")
            ]
            primary_window = matched_windows[0] if matched_windows else (enriched_failures[0]["real_defect_video_timestamp"] if enriched_failures else "")

            if primary_window and primary_window != "Whole Clip" and "Whole Clip" not in primary_window:
                if not ts_range or "whole clip" in ts_range.lower() or f"{duration:.1f}s" in ts_range:
                    sug["timestamp_range"] = primary_window
                else:
                    sug["timestamp_range"] = ts_range
            else:
                if not ts_range or "Whole Clip" in ts_range:
                    sug["timestamp_range"] = f"0.0s - {duration:.1f}s (Whole Clip)"
                else:
                    sug["timestamp_range"] = ts_range

            act_keywords = [
                "action", "walk", "run", "jump", "kick", "draw", "hold", "move", "pan", "tilt", "dolly",
                "zoom", "camera", "disappear", "appear", "vanish", "hand", "arm", "sword", "foot", "leg",
                "contact", "touch", "body", "pose", "turn", "face", "fall", "climb", "speed", "direction",
                "discontinuity", "morph", "teleport", "clip"
            ]
            text_to_check = (sug.get("issue", "") + " " + sug.get("tweak_instruction", "")).lower()
            rel_has_action = any(
                ef.get("claim_type") in ("action", "direction", "relative_position", "relative_size", "count", "physics_sanity", "existence")
                for ef in enriched_failures if ef.get("claim_id") in rel
            )
            if any(kw in text_to_check for kw in act_keywords) or rel_has_action:
                sug["fix_mode"] = "reshoot"
        return data
    except Exception as e:
        print(f"[PromptDirector] Failed to parse suggest_tweaks response: {e}")
        return {"suggestions": []}

def infer_tweak_axis(tweak_text: str) -> tuple[str, float]:
    """Classifies a user's tweak instruction into one of the 5 creative dimensions with a confidence score."""
    try:
        model = settings.DEFAULT_GEMINI_MODEL
        schema = {
            "type": "object",
            "properties": {
                "axis": {
                    "type": "string",
                    "enum": ["shot_framing_and_motion", "action", "location", "lighting", "style"]
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0-1.0. Below 0.6 means the instruction is ambiguous, vague, or spans multiple dimensions."
                }
            },
            "required": ["axis", "confidence"]
        }
        system_instruction = """You classify video tweak instructions into one of 5 filmmaking dimensions:
1. action (character movement, speed, gestures, physical actions)
2. shot_framing_and_motion (camera distance, angle, lens, camera motion like push-in/pan)
3. location (setting, background objects, environment, weather)
4. lighting (key/fill/backlight, brightness, shadows, darkness, color temperature, time of day)
5. style (visual tone, film grain, aesthetic, color grading)

If the instruction is ambiguous, too vague, purely subjective, or spans multiple dimensions (e.g. 'it feels weird', 'fix this', 'change everything'), assign a confidence score below 0.6.
"""
        def _execute():
            client = settings.get_genai_client()
            return client.models.generate_content(
                model=model,
                contents=[f"Tweak instruction: {tweak_text}"],
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0
                )
            )
        response = _call_with_retry(_execute)
        data = json.loads(response.text)
        axis = data.get("axis", "unknown")
        confidence = float(data.get("confidence", 0.0))
        return (axis, confidence)
    except Exception as e:
        print(f"[infer_tweak_axis ERROR] Failed to infer tweak axis: {e}")
        return ("unknown", 0.0)


