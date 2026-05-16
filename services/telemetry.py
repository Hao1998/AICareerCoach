import uuid

from flask import g, request


def init_request_id(app):
    @app.before_request
    def inject_request_id():
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))


def init_telemetry(app):
    endpoint = app.config.get('OTEL_EXPORTER_ENDPOINT')
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanExporter(exporter))
    trace.set_tracer_provider(provider)

    FlaskInstrumentor().instrument_app(app)
    SQLAlchemyInstrumentor().instrument(engine=db_engine(app))
    RedisInstrumentor().instrument()
    RequestsInstrumentor().instrument()


def db_engine(app):
    try:
        return app.extensions['sqlalchemy'].engine
    except (KeyError, AttributeError):
        return None


def init_sentry(app):
    dsn = app.config.get('SENTRY_DSN')
    if not dsn:
        return

    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration

    sentry_sdk.init(
        dsn=dsn,
        integrations=[FlaskIntegration()],
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
    )
