import sys
import os

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Dict, Any, Optional, List
import vertexai
from google import genai
from config.settings import settings
from engine.claims import extract_claims
from engine.pipeline import run_pipeline
from agents.conformal_judge import ConformalJudge
from agents.remediator import PromptRemediatorAgent

class CineQAAgentEngine:
    """
    Google Cloud Vertex AI Reasoning Engine / Agent Engine for AI Cinema Production.
    Complies with Vertex AI Agent Platform specifications.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
        confidence_level_alpha: float = 0.10
    ):
        self.project_id = project_id or settings.GOOGLE_CLOUD_PROJECT
        self.location = location
        self.model_name = model_name
        self.confidence_level_alpha = confidence_level_alpha
        self.conformal_judge = None
        self.remediator = None

    def set_up(self):
        """
        Initializes cloud clients, models, and conformal calibration on Vertex AI container.
        """
        vertexai.init(project=self.project_id, location=self.location)
        self.conformal_judge = ConformalJudge()
        self.remediator = PromptRemediatorAgent()
        print(f"[AgentEngine] Initialized CineQA Engine on Vertex AI (Project: {self.project_id}, Region: {self.location})")

    def extract_claims(self, scene_text: str) -> List[Dict[str, Any]]:
        """
        Extracts 7 atomic categories of visually verifiable claims from director scene prompt.
        """
        return extract_claims(scene_text, project=self.project_id, location=self.location)

    def verify_video(
        self, 
        scene_text: str, 
        video_path: str, 
        frames_dir: str = "temp_eval/frames",
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Runs adaptive frame extraction and 5-step causal reasoning consensus verification.
        """
        return run_pipeline(
            scene_text=scene_text,
            video_path=video_path,
            frames_dir=frames_dir,
            dry_run=dry_run,
            project=self.project_id,
            location=self.location
        )

    def remediate_prompt(self, scene_text: str, ledger: List[Dict[str, Any]], pass_rate: float) -> Dict[str, Any]:
        """
        Synthesizes targeted negative prompts, parameter adjustments, and localized inpaint intervals.
        """
        if not self.remediator:
            self.remediator = PromptRemediatorAgent()
        return self.remediator.remediate(scene_text, {"ledger": ledger, "pass_rate": pass_rate})

    def query(self, scene_text: str, video_path: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Full autonomous pipeline query method for Vertex AI Agent Engine.
        """
        if not self.conformal_judge:
            self.set_up()

        claims = self.extract_claims(scene_text)
        ledger = []
        rem_plan = None
        pass_rate = 0.0

        if video_path:
            ledger = self.verify_video(scene_text, video_path, dry_run=dry_run)
            total = len(ledger)
            matches = sum(1 for r in ledger if r.get("verdict") == "MATCH")
            pass_rate = (matches / total * 100) if total > 0 else 0.0

            if pass_rate < 75.0 or any(r.get("verdict") == "MISMATCH" for r in ledger):
                rem_plan = self.remediate_prompt(scene_text, ledger, pass_rate)

        return {
            "scene_text": scene_text,
            "claims": claims,
            "verification_ledger": ledger,
            "adherence_pass_rate": pass_rate,
            "remediation_plan": rem_plan,
            "status": "SUCCESS"
        }
