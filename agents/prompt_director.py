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
    client = settings.get_genai_client()
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
    client = settings.get_genai_client()
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

def suggest_tweaks(ledger: list[dict], director_choices: dict, final_prompt: str) -> dict:
    """
    Analyzes the entire verification ledger to produce deduplicated, prioritized
    conversational tweak suggestions for Gemini Omni video incremental editing.
    """
    failed_claims = [e for e in (ledger or []) if e.get("verdict") in ("MISMATCH", "CANNOT_DETERMINE")]
    if not failed_claims:
        return {"suggestions": []}

    client = settings.get_genai_client()
    model = settings.DEFAULT_GEMINI_MODEL

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
                            "description": "Clear explanation of the defect or inconsistency in English."
                        },
                        "tweak_instruction": {
                            "type": "string",
                            "description": "A single concise imperative sentence in English directly instructing Omni on the delta fix (e.g. 'Make the broken hole in the door persist and stay visible as the door swings open')."
                        },
                        "related_claims": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of claim_ids or claim types that were grouped together into this issue."
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Severity of this defect: high (critical physics/action breaks), medium (styling/lighting inconsistency), low (potential intentional choice or minor artifact)."
                        }
                    },
                    "required": ["issue", "tweak_instruction", "related_claims", "severity"]
                }
            }
        },
        "required": ["suggestions"]
    }

    system_instruction = """You are a senior VFX Supervisor and AI Cinematography Director.
You evaluate the verification ledger of a generated video take and formulate targeted, conversational tweak suggestions for Gemini Omni.

CRITICAL RULES:
1. MERGE ROOT-CAUSE ISSUES: Group together failures from physics_sanity and specific state/action claims if they refer to the exact same visual defect. Never output duplicate suggestions.
2. DELTA-ONLY IMPERATIVE INSTRUCTION: Output exactly ONE concise imperative sentence in English for `tweak_instruction` (e.g. "Make the broken hole in the door persist and stay visible as the door swings open"). DO NOT rewrite the whole prompt. Omni needs an incremental delta instruction.
3. LANGUAGE: ALL fields (`issue`, `tweak_instruction`, etc.) MUST be strictly in English.
4. RESPECT DIRECTOR CHOICES: Check `director_choices` (user's deliberate creative decisions). If a verification failure was actually caused by an intentional creative choice (e.g. user chose handheld camera, but verification flagged camera instability), DO NOT generate a fix, or mark it severity: "low" explaining it might be intentional.
5. SORT BY SEVERITY: Rank high severity (breaking physics, disappearing objects, wrong actions) first, followed by medium and low.
6. QUANTITY: Provide at most 3 to 4 suggestions. Prefer fewer, highly accurate suggestions over many vague ones.
"""

    prompt_content = f"""Final Shot Prompt: {final_prompt}

Director's Prior Guided Choices:
{json.dumps(director_choices or {}, ensure_ascii=False, indent=2)}

Verification Ledger:
{json.dumps(ledger or [], ensure_ascii=False, indent=2)}

Analyze the verification failures, deduplicate root causes, and produce 1-4 targeted conversational tweak instructions."""

    def _execute():
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
        return json.loads(response.text)
    except Exception as e:
        print(f"[PromptDirector] Failed to parse suggest_tweaks response: {e}")
        return {"suggestions": []}

def infer_tweak_axis(tweak_text: str) -> tuple[str, float]:
    """Classifies a user's tweak instruction into one of the 5 creative dimensions with a confidence score."""
    try:
        client = settings.get_genai_client()
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


