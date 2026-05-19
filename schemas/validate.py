"""
Request validation decorator.

`@validate_json(SchemaClass)` parses the incoming JSON body against a Pydantic
schema and passes the validated model as a `validated` keyword argument to the
view. Validation failures return a 400 with a structured error payload.
"""

from functools import wraps

from flask import jsonify, request
from pydantic import ValidationError


def validate_json(schema_class, allow_empty: bool = False):
    """Decorator that validates request JSON against a Pydantic schema.

    Args:
        schema_class: A Pydantic model class to validate against.
        allow_empty: If True, a missing or empty body is treated as `{}` —
            useful when all schema fields have defaults.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.get_json(silent=True)
            if data is None:
                if not allow_empty:
                    return jsonify({"success": False, "error": "Request body must be JSON"}), 400
                data = {}
            try:
                validated = schema_class(**data)
            except ValidationError as e:
                details = []
                for err in e.errors():
                    field = '.'.join(str(loc) for loc in err['loc']) or '<root>'
                    details.append(f"{field}: {err['msg']}")
                return jsonify({
                    "success": False,
                    "error": "Validation failed",
                    "details": details,
                }), 400

            kwargs['validated'] = validated
            return f(*args, **kwargs)
        return wrapper
    return decorator
