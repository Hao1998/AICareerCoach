"""
WebSocket Chat Controller

Handles real-time chat via Socket.IO. Replaces the SSE-based /api/chat/stream.
Events are routed through Redis so all gunicorn workers see every message.
"""

import json
import logging
import threading
from datetime import datetime

from flask import request
from flask_login import current_user
from flask_socketio import emit, disconnect

from factory import socketio
from models import db, ChatMessage
from services.db_lock import safe_commit
from services.input_guard import scan_message
from services.streaming import acquire_stream_slot, release_stream_slot, WebSocketStreamHandler

logger = logging.getLogger(__name__)

_cancel_events: dict[int, threading.Event] = {}


@socketio.on('connect')
def handle_connect():
    if not current_user.is_authenticated:
        disconnect()
        return
    logger.info("WebSocket connected: user=%s sid=%s", current_user.id, request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    if current_user.is_authenticated:
        cancel_evt = _cancel_events.pop(current_user.id, None)
        if cancel_evt:
            cancel_evt.set()
        logger.info("WebSocket disconnected: user=%s", current_user.id)


@socketio.on('chat_message')
def handle_chat_message(data):
    if not current_user.is_authenticated:
        disconnect()
        return

    message = (data.get('message') or '').strip()
    if not message:
        emit('error', {'error': 'Message is required'})
        return
    if len(message) > 2000:
        emit('error', {'error': 'Message too long (max 2000 characters)'})
        return

    guard = scan_message(message)
    if not guard['safe']:
        emit('token', {'content': guard['response']})
        emit('done', {})
        return

    user_id = current_user.id

    if not acquire_stream_slot(user_id):
        emit('error', {'error': 'You already have a message in flight. Wait for it to finish.'})
        return

    cancel_event = threading.Event()
    _cancel_events[user_id] = cancel_event

    from flask import current_app
    app = current_app._get_current_object()
    sid = request.sid

    def runner():
        try:
            from chatbot import CareerCoachChatbot
            chatbot = CareerCoachChatbot(app)
            handler = WebSocketStreamHandler(sid, cancel_event)

            with app.app_context():
                from chatbot.agent import (
                    detect_session_boundary, _load_context, build_system_prompt,
                    get_conversation_history, build_tools, _build_agent, _invoke_agent,
                    _extract_intent, _tool_steps_from_messages, stream_fallback_text,
                )
                from services.llm_service import get_streaming_llm

                if detect_session_boundary(app, user_id):
                    threading.Thread(
                        target=chatbot._summarize_in_background,
                        args=(user_id,),
                        daemon=True,
                    ).start()

                db.session.add(ChatMessage(
                    user_id=user_id, role='user',
                    content=message, timestamp=datetime.utcnow(),
                ))
                safe_commit()

                user, resume, config, liked_count, disliked_count = _load_context(app, user_id)
                llm = get_streaming_llm(app)
                system_prompt = build_system_prompt(user, resume, config, liked_count, disliked_count)
                chat_history = get_conversation_history(user_id, limit=10)
                if chat_history:
                    chat_history = chat_history[:-1]

                tools = build_tools(app, user_id, progress_cb=handler.push_progress)
                agent = _build_agent(llm, tools, system_prompt)

                response_text, messages = _invoke_agent(
                    agent, message, chat_history, callbacks=[handler]
                )

                if cancel_event.is_set():
                    return

                # If nothing streamed (e.g. cached response), push the final text
                # so the UI shows it instead of "(no response)".
                fallback = stream_fallback_text(handler.captured_text, response_text)
                if fallback:
                    socketio.emit('token', {'content': fallback}, room=sid)

                response_text = handler.captured_text or response_text or (
                    "I'm sorry, I couldn't process your request."
                )
                intent, action_data = _extract_intent(_tool_steps_from_messages(messages))

                db.session.add(ChatMessage(
                    user_id=user_id, role='assistant',
                    content=response_text, timestamp=datetime.utcnow(),
                    intent=intent, action_data=action_data,
                ))
                safe_commit()

                socketio.emit('done', {
                    'intent': intent,
                    'action_data': json.loads(action_data) if action_data else None,
                }, room=sid)

        except Exception as exc:
            logger.exception("WebSocket chat failed for user %s", user_id)
            socketio.emit('error', {'error': str(exc)}, room=sid)
        finally:
            release_stream_slot(user_id)
            _cancel_events.pop(user_id, None)

    threading.Thread(target=runner, daemon=True).start()


@socketio.on('cancel_stream')
def handle_cancel():
    if not current_user.is_authenticated:
        return
    cancel_evt = _cancel_events.get(current_user.id)
    if cancel_evt:
        cancel_evt.set()
    release_stream_slot(current_user.id)
