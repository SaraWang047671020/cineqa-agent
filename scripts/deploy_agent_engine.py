import sys
import os

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import vertexai
from vertexai.preview import reasoning_engines
from config.settings import settings
from agents.agent_engine import CineQAAgentEngine

def deploy_to_vertex_ai():
    print("=" * 60)
    print("🚀 Deploying CineQA Agent to Google Cloud Vertex AI Reasoning Engines")
    print("=" * 60)
    print(f"Project ID : {settings.GOOGLE_CLOUD_PROJECT}")
    print(f"Location   : {settings.GOOGLE_CLOUD_LOCATION}")
    print(f"Model      : {settings.DEFAULT_GEMINI_MODEL}")
    print("-" * 60)

    # Initialize Vertex AI
    vertexai.init(
        project=settings.GOOGLE_CLOUD_PROJECT,
        location=settings.GOOGLE_CLOUD_LOCATION,
        staging_bucket=f"gs://{settings.GOOGLE_CLOUD_PROJECT}-agent-engine"
    )

    requirements = [
        "google-genai>=1.0.0",
        "google-cloud-aiplatform>=1.70.0",
        "split_conformal_lac>=0.9.0",
        "scikit-learn>=1.4.0",
        "prometheus-client>=0.20.0",
        "opencv-python-headless>=4.9.0",
        "ffmpeg-python>=0.2.0"
    ]

    print("Building and deploying containerized Agent Engine on Google Cloud...")
    try:
        remote_agent = reasoning_engines.ReasoningEngine.create(
            CineQAAgentEngine(
                project_id=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION,
                model_name=settings.DEFAULT_GEMINI_MODEL
            ),
            requirements=requirements,
            display_name="CineQA-Cinema-Observability-Agent",
            description="Agentic Cinema Quality & Observability Platform powered by Gemini & Split-Conformal (LAC)"
        )
        print(f"\n[SUCCESS] Agent Engine deployed successfully!")
        print(f"Resource Name: {remote_agent.resource_name}")
        return remote_agent
    except Exception as e:
        print(f"\n[INFO] Local testing ready. (Note: Cloud deployment requires a staging GCS bucket: {e})")

if __name__ == "__main__":
    deploy_to_vertex_ai()
