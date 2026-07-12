"""
Chat Controller

Handles Career Coach AI chatbot endpoints and the chat widget.
Blueprint: 'chat'
"""

import logging

from flask import Blueprint, request, render_template, jsonify, current_app
from flask_login import login_required, current_user
from flask_limiter.errors import RateLimitExceeded

from factory import limiter
from models import db, ChatMessage
from services.db_lock import safe_commit
from services.input_guard import scan_message
from schemas.request_schemas import ChatMessageRequest
from schemas.validate import validate_json

chat_bp = Blueprint('chat', __name__)

logger = logging.getLogger(__name__)


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
@validate_json(ChatMessageRequest)
def chat_api(validated: ChatMessageRequest):
    try:
        message = validated.message

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
    except Exception:
        logger.exception("Chat request failed for user %s", current_user.id)
        return jsonify({"success": False, "error": "Something went wrong processing your message. Please try again."}), 500


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
    safe_commit()
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
