import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def _get_secret_val(key_names: list[str], default: str = "") -> str:
    """
    Robust secret lookup supporting os.environ, st.session_state, and
    hierarchical/flat st.secrets on Streamlit Cloud.
    """
    # 1. os.environ
    for k in key_names:
        val = os.getenv(k)
        if val:
            return val

    # 2. st.session_state (runtime user overrides)
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            for k in key_names:
                v = st.session_state.get(k.lower()) or st.session_state.get(k)
                if v:
                    return str(v).strip()
    except Exception:
        pass

    # 3. st.secrets (flat & nested)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k in key_names:
                if k in st.secrets:
                    return str(st.secrets[k]).strip()
                if k.lower() in st.secrets:
                    return str(st.secrets[k.lower()]).strip()
            # Recursive check in sections (e.g. [general], [gemini], [gcp])
            def _walk(d):
                for dk, dv in d.items():
                    if dk.lower() in [x.lower() for x in key_names] and isinstance(dv, (str, int, float, bool)):
                        return str(dv).strip()
                    if isinstance(dv, dict) or hasattr(dv, "items"):
                        res = _walk(dict(dv))
                        if res:
                            return res
                return None
            found = _walk(dict(st.secrets))
            if found:
                return found
    except Exception:
        pass

    return default


class Settings:
    _clients: dict = {}
    _lock = None
    _force_disable_vertex: bool = False

    # Observability & Split-Conformal (LAC) Settings
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    CONFIDENCE_LEVEL_ALPHA: float = float(os.getenv("CONFIDENCE_LEVEL_ALPHA", "0.10"))  # 90% confidence
    HIGH_UNCERTAINTY_THRESHOLD: float = float(os.getenv("HIGH_UNCERTAINTY_THRESHOLD", "20.0"))
    DEFAULT_GEMINI_MODEL: str = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.6-flash")

    @property
    def GEMINI_API_KEY(self) -> str:
        return _get_secret_val(["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])

    @property
    def GOOGLE_CLOUD_PROJECT(self) -> str:
        return _get_secret_val(["GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "GCLOUD_PROJECT", "project_id"])

    @property
    def GOOGLE_CLOUD_LOCATION(self) -> str:
        return _get_secret_val(["GOOGLE_CLOUD_LOCATION", "GCP_LOCATION", "location"], default="global")

    @property
    def USE_VERTEX_AI(self) -> bool:
        if self._force_disable_vertex:
            return False
        val = _get_secret_val(["USE_VERTEX_AI", "use_vertex_ai"], default="false").lower()
        return val in ("true", "1", "yes")

    def force_disable_vertex(self):
        """Forces fallback to Gemini API Key and flushes client cache."""
        self._force_disable_vertex = True
        self._clients.clear()

    def get_genai_client(self, location_override: str = None) -> genai.Client:
        """
        Automatically initializes and caches the GenAI Client based on environment.
        Guarantees ZERO hangs on http://metadata.google.internal by never invoking
        google.auth.default() when GEMINI_API_KEY is available or when running outside GCE.
        """
        import threading
        if self._lock is None:
            self._lock = threading.Lock()

        loc = location_override or self.GOOGLE_CLOUD_LOCATION

        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not (sa_path and os.path.exists(sa_path)):
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
                    import tempfile, json
                    key_dict = dict(st.secrets["gcp_service_account"])
                    key_path = os.path.join(tempfile.gettempdir(), "gcp_key.json")
                    if not os.path.exists(key_path):
                        with open(key_path, "w") as f:
                            json.dump(key_dict, f)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
                    sa_path = key_path
            except Exception:
                pass

        sa_key = _get_secret_val(["GCP_SERVICE_ACCOUNT_KEY", "gcp_service_account_key"])
        has_explicit_sa = bool((sa_path and os.path.exists(sa_path)) or sa_key)

        # Only use Vertex AI if explicitly requested AND an explicit service account key is available,
        # OR if no Gemini API Key is provided at all and explicit SA exists.
        use_vertex = False
        if self.USE_VERTEX_AI and self.GOOGLE_CLOUD_PROJECT and has_explicit_sa:
            use_vertex = True
        elif not self.GEMINI_API_KEY and self.GOOGLE_CLOUD_PROJECT and has_explicit_sa:
            use_vertex = True

        cache_key = (use_vertex, self.GOOGLE_CLOUD_PROJECT, loc, self.GEMINI_API_KEY)

        with self._lock:
            if cache_key in self._clients:
                return self._clients[cache_key]

            client = None
            if use_vertex:
                try:
                    print(f"[CineQA] Initializing Google GenAI client via Vertex AI (Project: {self.GOOGLE_CLOUD_PROJECT}, Location: {loc})")
                    client = genai.Client(
                        vertexai=True,
                        project=self.GOOGLE_CLOUD_PROJECT,
                        location=loc
                    )
                except Exception as e:
                    print(f"[CineQA] Vertex AI client init failed: {e}. Falling back to Gemini API Key...")
                    client = None

            if client is None:
                if self.GEMINI_API_KEY:
                    print("[CineQA] Initializing Google GenAI client via Gemini API Key")
                    client = genai.Client(api_key=self.GEMINI_API_KEY)
                else:
                    print("[CineQA] Warning: No GEMINI_API_KEY found; attempting default client")
                    client = genai.Client()

            self._clients[cache_key] = client
            return client

settings = Settings()
