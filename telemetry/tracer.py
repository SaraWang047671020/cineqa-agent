from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource

import os

resource = Resource.create({"service.name": "cineqa-agent"})
provider = TracerProvider(resource=resource)
# For local development, export to console/memory (disabled by default to avoid terminal spam)
if os.environ.get("DEBUG_TRACING") == "1":
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("cineqa.tracer")
