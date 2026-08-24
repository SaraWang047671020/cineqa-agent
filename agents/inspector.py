import json
import time
import numpy as np
from google import genai
from google.genai import types
from telemetry.tracer import tracer
from telemetry.metrics import (
    PROMPT_ALIGNMENT_SCORE, 
    TAKES_TOTAL, 
    INSPECTION_DURATION_SECONDS
)
from agents.conformal_judge import ConformalJudge
from config.settings import settings

class VideoInspectorAgent:
    """
    Inspects generated video against structured claims using Gemini 2.0 + MAPIE Conformal Judge.
    """
    def __init__(self, client: genai.Client = None):
        self.client = client or settings.get_genai_client()
        self.conformal_judge = ConformalJudge()

    def inspect(self, video_path: str, claims: dict, shot_id: str = "shot_001") -> dict:
        start_time = time.time()
        with tracer.start_as_current_span("VideoInspector.inspect"):
            # Upload video file to Gemini API / Vertex AI Files
            uploaded_file = self.client.files.upload(file=video_path)

            prompt = f"""
            You are a strict Cinematic Quality Control Supervisor (VFX & Script Supervisor).
            Analyze the uploaded video frame-by-frame against the following structured requirements:
            {json.dumps(claims, indent=2)}

            Output a valid JSON object matching this schema:
            {{
                "camera_alignment_score": (int 0-100),
                "subject_action_score": (int 0-100),
                "lighting_environment_score": (int 0-100),
                "overall_raw_score": (int 0-100),
                "timeline_defects": [
                    {{"timestamp": "00:02", "type": "camera_drift|limb_melting|prop_disappearance", "severity": "minor|major|critical", "description": "..."}}
                ],
                "summary": "Concise summary of video-prompt alignment"
            }}
            """

            response = self.client.models.generate_content(
                model=settings.DEFAULT_GEMINI_MODEL,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )

            result = json.loads(response.text)
            elapsed = time.time() - start_time
            INSPECTION_DURATION_SECONDS.observe(elapsed)

            # Record raw scores
            PROMPT_ALIGNMENT_SCORE.labels(shot_id=shot_id, dimension='camera').set(result['camera_alignment_score'])
            PROMPT_ALIGNMENT_SCORE.labels(shot_id=shot_id, dimension='action').set(result['subject_action_score'])
            PROMPT_ALIGNMENT_SCORE.labels(shot_id=shot_id, dimension='lighting').set(result['lighting_environment_score'])

            # Evaluate with MAPIE Conformal Uncertainty
            num_defects = len(result.get("timeline_defects", []))
            feature_vector = np.array([float(result["overall_raw_score"]), 100.0, 0.5, float(num_defects)])
            conformal_res = self.conformal_judge.evaluate_with_intervals(
                raw_score=float(result["overall_raw_score"]),
                features=feature_vector,
                shot_id=shot_id,
                dimension="overall"
            )

            result["conformal_analysis"] = conformal_res

            # Record take status
            if conformal_res["decision"] == "AUTO_PASS":
                TAKES_TOTAL.labels(status="passed", defect_type="none").inc()
            elif conformal_res["decision"] == "AUTO_REMEDIATE":
                top_defect = result["timeline_defects"][0]["type"] if result["timeline_defects"] else "alignment_mismatch"
                TAKES_TOTAL.labels(status="failed", defect_type=top_defect).inc()

            return result
