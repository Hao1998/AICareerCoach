"""
Chat Controller

Handles Career Coach AI chatbot endpoints and the chat widget.
Blueprint: 'chat'
"""

import json
import queue
import threading

from flask import Blueprint, request, render_template, jsonify, current_app, Response, stream_with_context
from flask_login import login_required, current_user
from flask_limiter.errors import RateLimitExceeded

from factory import limiter
from models import db, ChatMessage
from services.input_guard import scan_message
from services.streaming import acquire_stream_slot, release_stream_slot

chat_bp = Blueprint('chat', __name__)


@chat_bp.errorhandler(RateLimitExceeded)
def handle_rate_limit(e):
    return jsonify({"success": False, "error": "Too many messages. Please slow down."}), 429

_chatbot = None


def _get_chatbot():
    """Lazy-initialize the chatbot (imported here to avoid module-load-time issues)"""
    global _chatbot
    if _chatbot is None:
        from chatbot import CareerCoachChatbot
        _chatbot = CareerCoachChatbot(current_app._get_current_object())
    return _chatbot


@chat_bp.route('/api/chat', methods=['POST'])
@login_required
@limiter.limit("20 per minute; 200 per day")
def chat_api():
    try:
        data = request.get_json()
        if not data or not data.get('message', '').strip():
            return jsonify({"success": False, "error": "Message is required"}), 400

        message = data['message'].strip()
        if len(message) > 2000:
            return jsonify({"success": False, "error": "Message too long (max 2000 characters)"}), 400

        guard = scan_message(message)
        if not guard["safe"]:
            return jsonify({"success": True, "response": guard["response"]}), 200

        result = _get_chatbot().chat(current_user.id, message)

        return jsonify({
            "success": True,
            "response": result["response"],
            "intent": result.get("intent"),
            "action_data": result.get("action_data")
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Chat error: {str(e)}"}), 500


@chat_bp.route('/api/chat/stream', methods=['POST'])
@login_required
@limiter.limit("20 per minute; 200 per day")
def chat_stream_api():
    """
    Server-Sent Events streaming variant of /api/chat.

    Wire format (each line is a separate SSE event):
        data: {"type":"token","content":"Hello"}\n\n
        data: {"type":"tool_start","tool":"find_top_jobs"}\n\n
        data: {"type":"tool_end"}\n\n
        data: {"type":"done","intent":"...","action_data":{...}}\n\n

    Rate limits applied:
      • Flask-Limiter:    20 connections/min, 200/day per user (existing).
      • Concurrent guard: max 1 in-flight stream per user — second
                          request gets HTTP 409 immediately.
    """
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({"success": False, "error": "Message is required"}), 400
    if len(message) > 2000:
        return jsonify({"success": False, "error": "Message too long (max 2000 characters)"}), 400

    guard = scan_message(message)
    if not guard["safe"]:
        # Synthetic single-event stream so the frontend code path is uniform.
        def guarded():
            yield f"data: {json.dumps({'type': 'token', 'content': guard['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return Response(guarded(), mimetype='text/event-stream')

    user_id = current_user.id

    # Concurrent stream limit (Layer 2 of rate limiting).
    if not acquire_stream_slot(user_id):
        return jsonify({
            "success": False,
            "error": "You already have a message in flight. Wait for it to finish.",
        }), 409

    event_queue: queue.Queue = queue.Queue()
    app = current_app._get_current_object()

    # Lazy-load the chatbot (same pattern as the non-streaming endpoint).
    from chatbot import CareerCoachChatbot
    chatbot = CareerCoachChatbot(app)

    # Run the agent in a background thread so this request thread is free to
    # drain the queue and yield SSE events.
    def runner():
        try:
            chatbot.chat_stream(user_id, message, event_queue)
        finally:
            release_stream_slot(user_id)

    threading.Thread(target=runner, daemon=True).start()

    @stream_with_context
    def event_stream():
        # Initial comment line keeps proxies (nginx, ALB) from buffering before
        # the first real event arrives.
        yield ": ready\n\n"
        while True:
            event = event_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',   # disable nginx buffering for true streaming
            'Connection': 'keep-alive',
        },
    )


@chat_bp.route('/api/chat/history', methods=['GET'])
@login_required
def chat_history_api():
    limit = request.args.get('limit', 50, type=int)
    messages = ChatMessage.query.filter_by(
        user_id=current_user.id
    ).order_by(ChatMessage.timestamp.asc()).limit(limit).all()
    return jsonify({"messages": [m.to_dict() for m in messages]})


@chat_bp.route('/api/chat/history', methods=['DELETE'])
@login_required
def clear_chat_history():
    ChatMessage.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({"success": True})


@chat_bp.route('/chat-widget')
@login_required
def chat_widget():
    return render_template('chat_widget.html')


@chat_bp.after_app_request
def inject_chat_widget_script(response):
    """Inject chat widget loader script into all authenticated HTML responses"""
    if (not response.direct_passthrough and
            response.content_type and
            response.content_type.startswith('text/html') and
            current_user.is_authenticated and
            response.status_code == 200):
        script = b'<script src="/static/js/chat_widget_loader.js" defer></script></body>'
        response.data = response.data.replace(b'</body>', script)
    return response
