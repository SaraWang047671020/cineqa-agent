import json
from google import genai
from google.genai import types
from telemetry.tracer import tracer
from config.settings import settings

class PromptDecomposerAgent:
    """
    Decomposes natural language director prompts into structured verification claims.
    """
    def __init__(self, client: genai.Client = None):
        self.client = client or settings.get_genai_client()

    def decompose(self, raw_prompt: str) -> dict:
        with tracer.start_as_current_span("PromptDecomposer.decompose"):
            system_instruction = """
            You are an expert film Script Supervisor and AI Cinema Prompt Engineer.
            Decompose the given video generation prompt into a structured JSON verification schema.
            The JSON MUST have these exact top-level keys:
            {
                "camera": {"motion": "string", "angle": "string", "lens_lighting": "string"},
                "subject": {"character": "string", "costume": "string", "key_action": "string"},
                "environment": {"location": "string", "atmosphere": "string", "weather_time": "string"},
                "continuity_constraints": ["string list of critical details that must not drift"]
            }
            """
            response = self.client.models.generate_content(
                model=settings.DEFAULT_GEMINI_MODEL,
                contents=[raw_prompt, system_instruction],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            return json.loads(response.text)
