import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# If running on Streamlit Cloud, inject st.secrets into environment variables
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ.setdefault(k, str(v))
except Exception:
    pass

class Settings:
    # Google Cloud / Vertex AI Settings
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    USE_VERTEX_AI: bool = os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1", "yes")

    # Google AI Studio API Key
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model configuration
    DEFAULT_GEMINI_MODEL: str = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.6-flash")

    # Observability & Split-Conformal (LAC) Settings
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    CONFIDENCE_LEVEL_ALPHA: float = float(os.getenv("CONFIDENCE_LEVEL_ALPHA", "0.10"))  # 90% confidence
    HIGH_UNCERTAINTY_THRESHOLD: float = float(os.getenv("HIGH_UNCERTAINTY_THRESHOLD", "20.0"))

    _clients: dict = {}
    _lock = None

    def get_genai_client(self, location_override: str = None) -> genai.Client:
        """
        Automatically initializes and caches the GenAI Client based on environment.
        """
        import threading
        if self._lock is None:
            self._lock = threading.Lock()

        loc = location_override or self.GOOGLE_CLOUD_LOCATION
        cache_key = (self.USE_VERTEX_AI, self.GOOGLE_CLOUD_PROJECT, loc, self.GEMINI_API_KEY)
        
        with self._lock:
            if cache_key in self._clients:
                return self._clients[cache_key]

            if self.USE_VERTEX_AI or (self.GOOGLE_CLOUD_PROJECT and not self.GEMINI_API_KEY):
                print(f"[CineQA] Initializing Google GenAI client via Vertex AI (Project: {self.GOOGLE_CLOUD_PROJECT}, Location: {loc})")
                client = genai.Client(
                    vertexai=True,
                    project=self.GOOGLE_CLOUD_PROJECT,
                    location=loc
                )
            else:
                print("[CineQA] Initializing Google GenAI client via Gemini API Key")
                client = genai.Client(api_key=self.GEMINI_API_KEY)

            self._clients[cache_key] = client
            return client

settings = Settings()
