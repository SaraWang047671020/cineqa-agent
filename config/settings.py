import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class Settings:
    # Google Cloud / Vertex AI Settings
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    USE_VERTEX_AI: bool = os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1", "yes")

    # Google AI Studio API Key
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model configuration
    DEFAULT_GEMINI_MODEL: str = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.0-flash")

    # Observability & MAPIE Settings
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    CONFIDENCE_LEVEL_ALPHA: float = float(os.getenv("CONFIDENCE_LEVEL_ALPHA", "0.10"))  # 90% confidence
    HIGH_UNCERTAINTY_THRESHOLD: float = float(os.getenv("HIGH_UNCERTAINTY_THRESHOLD", "20.0"))

    def get_genai_client(self) -> genai.Client:
        """
        Automatically initializes the GenAI Client based on environment:
        - If USE_VERTEX_AI is True or GOOGLE_CLOUD_PROJECT is provided -> Uses Vertex AI / GCP Project
        - Else -> Uses GEMINI_API_KEY
        """
        if self.USE_VERTEX_AI or (self.GOOGLE_CLOUD_PROJECT and not self.GEMINI_API_KEY):
            print(f"[CineQA] Initializing Google GenAI client via Vertex AI (Project: {self.GOOGLE_CLOUD_PROJECT}, Location: {self.GOOGLE_CLOUD_LOCATION})")
            return genai.Client(
                vertexai=True,
                project=self.GOOGLE_CLOUD_PROJECT,
                location=self.GOOGLE_CLOUD_LOCATION
            )
        else:
            print("[CineQA] Initializing Google GenAI client via Gemini API Key")
            return genai.Client(api_key=self.GEMINI_API_KEY)

settings = Settings()
