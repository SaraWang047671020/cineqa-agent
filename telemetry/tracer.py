class DummySpan:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def record_exception(self, *args, **kwargs): pass

class DummyTracer:
    def start_as_current_span(self, name, *args, **kwargs):
        return DummySpan()

tracer = DummyTracer()
