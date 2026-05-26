# app/services/chat_service.py

import json
import uuid
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterator, Any

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS

logger = logging.getLogger(__name__)

CHATS_DATA_DIR.mkdir(parents=True, exist_ok=True)

VALID_ROLES = {"user", "assistant"}


# -------------------------------------------------------------------------
# MESSAGE MODEL
# -------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str


# -------------------------------------------------------------------------
# MAIN CHAT SERVICE
# -------------------------------------------------------------------------

class ChatService:

    def __init__(
        self,
        groq_service,
        realtime_service=None,
        brain_service=None,
        task_executor=None,
        vision_service=None,
        task_manager=None,
    ):
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        self.brain_service = brain_service
        self.task_executor = task_executor
        self.vision_service = vision_service
        self.task_manager = task_manager

        # session_id -> List[ChatMessage]
        self.sessions: Dict[str, List[ChatMessage]] = {}

    # ---------------------------------------------------------------------
    # SESSION HELPERS
    # ---------------------------------------------------------------------

    def _sanitize_session_id(self, session_id: str) -> str:
        return session_id.replace("-", "").replace(" ", "_")

    def validate_session_id(self, session_id: str) -> bool:

        if not session_id or not session_id.strip():
            return False

        if ".." in session_id:
            return False

        if "/" in session_id or "\\" in session_id:
            return False

        if len(session_id) > 255:
            return False

        return True

    def get_session_filepath(self, session_id: str):
        safe_session_id = self._sanitize_session_id(session_id)
        filename = f"chat_{safe_session_id}.json"
        return CHATS_DATA_DIR / filename

    # ---------------------------------------------------------------------
    # SESSION LOAD
    # ---------------------------------------------------------------------

    def load_session_from_disk(self, session_id: str) -> bool:

        filepath = self.get_session_filepath(session_id)

        if not filepath.exists():
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)

            messages_data = chat_dict.get("messages", [])

            messages = []

            for msg in messages_data:

                role = msg.get("role")
                content = msg.get("content", "")

                if role not in VALID_ROLES:
                    continue

                messages.append(
                    ChatMessage(
                        role=role,
                        content=content
                    )
                )

            self.sessions[session_id] = messages

            logger.info("Loaded session: %s", session_id)

            return True

        except Exception as e:
            logger.warning(
                "Failed to load session %s: %s",
                session_id,
                e
            )
            return False

    # ---------------------------------------------------------------------
    # CREATE / GET SESSION
    # ---------------------------------------------------------------------

    def get_or_create_session(
        self,
        session_id: Optional[str] = None
    ) -> str:

        if not session_id:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = []
            return session_id

        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id: {session_id}"
            )

        if session_id in self.sessions:
            return session_id

        if self.load_session_from_disk(session_id):
            return session_id

        self.sessions[session_id] = []

        return session_id

    # ---------------------------------------------------------------------
    # MESSAGE MANAGEMENT
    # ---------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:

        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")

        content = content.strip()

        if not content:
            raise ValueError("Message content cannot be empty.")

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            ChatMessage(
                role=role,
                content=content
            )
        )

    def get_chat_history(
        self,
        session_id: str
    ) -> List[ChatMessage]:

        return self.sessions.get(session_id, [])

    # ---------------------------------------------------------------------
    # FORMAT HISTORY FOR LLM
    # ---------------------------------------------------------------------

    def format_history_for_llm(
        self,
        session_id: str,
        exclude_last: bool = False
    ) -> List[Tuple[str, str]]:

        messages = self.get_chat_history(session_id)

        if exclude_last and messages:
            messages = messages[:-1]

        history = []

        i = 0

        while i < len(messages) - 1:

            user_msg = messages[i]
            assistant_msg = messages[i + 1]

            if (
                user_msg.role == "user"
                and assistant_msg.role == "assistant"
            ):
                history.append(
                    (
                        user_msg.content,
                        assistant_msg.content
                    )
                )
                i += 2
            else:
                i += 1

        return history[-MAX_CHAT_HISTORY_TURNS:]

    # ---------------------------------------------------------------------
    # GENERAL CHAT
    # ---------------------------------------------------------------------

    def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> str:

        self.add_message(
            session_id,
            "user",
            user_message
        )

        chat_history = self.format_history_for_llm(
            session_id,
            exclude_last=True
        )

        response = self.groq_service.get_response(
            question=user_message,
            chat_history=chat_history
        )

        self.add_message(
            session_id,
            "assistant",
            response
        )

        self.save_chat_session(session_id)

        return response

    def process_message_stream(
        self,
        session_id: str,
        user_message: str
    ) -> Iterator[Any]:

        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(
            session_id,
            exclude_last=True
        )

        full_response = ""

        for chunk in self.groq_service.stream_response(
            question=user_message,
            chat_history=chat_history
        ):
            if isinstance(chunk, str):
                full_response += chunk
            yield chunk

        if full_response:
            self.add_message(session_id, "assistant", full_response)
            self.save_chat_session(session_id)

    # ---------------------------------------------------------------------
    # REALTIME CHAT
    # ---------------------------------------------------------------------

    def process_realtime_message(
        self,
        session_id: str,
        user_message: str
    ) -> str:

        if not self.realtime_service:
            raise ValueError(
                "Realtime service is not initialized."
            )

        self.add_message(
            session_id,
            "user",
            user_message
        )

        chat_history = self.format_history_for_llm(
            session_id,
            exclude_last=True
        )

        response = self.realtime_service.get_response(
            question=user_message,
            chat_history=chat_history
        )

        self.add_message(
            session_id,
            "assistant",
            response
        )

        self.save_chat_session(session_id)

        return response

    def process_realtime_message_stream(
        self,
        session_id: str,
        user_message: str
    ) -> Iterator[Any]:

        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized.")

        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(
            session_id,
            exclude_last=True
        )

        full_response = ""

        for chunk in self.realtime_service.stream_response(
            question=user_message,
            chat_history=chat_history
        ):
            if isinstance(chunk, str):
                full_response += chunk
            yield chunk

        if full_response:
            self.add_message(session_id, "assistant", full_response)
            self.save_chat_session(session_id)

    # ---------------------------------------------------------------------
    # JARVIS UNIFIED STREAM (brain -> classify -> route -> execute/stream)
    # ---------------------------------------------------------------------

    def process_jarvis_message_stream(
        self,
        session_id: str,
        user_message: str,
        imgbase64: Optional[str] = None,
    ) -> Iterator[Any]:
        from app.services.decision_types import (
            CATEGORY_GENERAL, CATEGORY_REALTIME, CATEGORY_CAMERA,
            CATEGORY_TASK, CATEGORY_MIXED,
            HEAVY_INTENTS, INSTANT_INTENTS,
        )

        chat_history_pairs = self.format_history_for_llm(session_id)

        # --- Vision / camera path ---
        if imgbase64 and self.vision_service:
            self.add_message(session_id, "user", user_message)
            yield {"_activity": [{"event": "analyzing_image", "message": "Analyzing camera image…"}]}
            vision_result = self.vision_service.describe_image(imgbase64, prompt=user_message)
            self.add_message(session_id, "assistant", vision_result)
            self.save_chat_session(session_id)
            yield vision_result
            return

        # --- Brain classification ---
        category = CATEGORY_GENERAL
        task_types: List[str] = []

        if self.brain_service:
            category, task_types, method, ms = self.brain_service.classify(
                user_message, chat_history_pairs
            )
            yield {"_activity": [{"event": "classified", "message": f"Category: {category} ({method}, {ms}ms)"}]}

        self.add_message(session_id, "user", user_message)

        # --- Camera (no image provided) ---
        if category == CATEGORY_CAMERA:
            response = "Please turn on your camera so I can see what you're referring to."
            self.add_message(session_id, "assistant", response)
            self.save_chat_session(session_id)
            yield response
            return

        # --- Task path ---
        if category in (CATEGORY_TASK, CATEGORY_MIXED) and self.brain_service and self.task_executor:
            intents = self.brain_service.extract_task_payloads(
                user_message, task_types, chat_history_pairs
            )
            yield {"_activity": [{"event": "executing_tasks", "message": f"Executing {len(intents)} task(s)…"}]}

            heavy = [(i, p) for i, p in intents if i in HEAVY_INTENTS]
            instant = [(i, p) for i, p in intents if i in INSTANT_INTENTS]

            # Submit heavy tasks to background
            bg_task_ids = []
            if heavy and self.task_manager:
                for intent_type, payload in heavy:
                    tid = self.task_manager.submit(intent_type, payload, chat_history_pairs)
                    bg_task_ids.append({"task_id": tid, "intent": intent_type})

            if bg_task_ids:
                yield {"_background_tasks": bg_task_ids}

            # Execute instant tasks immediately
            if instant:
                result = self.task_executor.execute(instant, chat_history_pairs)
                actions: dict = {}
                if result.wopens:
                    actions["wopens"] = result.wopens
                if result.plays:
                    actions["plays"] = result.plays
                if result.images:
                    actions["images"] = [u for u, _ in result.images]
                if result.contents:
                    actions["contents"] = result.contents
                if result.googlesearches:
                    actions["googlesearches"] = result.googlesearches
                if result.youtubesearches:
                    actions["youtubesearches"] = result.youtubesearches
                if result.cam:
                    actions["cam"] = result.cam
                if actions:
                    yield {"_actions": actions}

            # For mixed category also stream a conversational reply
            if category == CATEGORY_MIXED or (not heavy and not instant):
                full_response = ""
                for chunk in self.realtime_service.stream_response(
                    question=user_message,
                    chat_history=chat_history_pairs
                ) if self.realtime_service else self.groq_service.stream_response(
                    question=user_message,
                    chat_history=chat_history_pairs
                ):
                    if isinstance(chunk, str):
                        full_response += chunk
                    yield chunk
                if full_response:
                    self.add_message(session_id, "assistant", full_response)
                    self.save_chat_session(session_id)
            else:
                # Tasks-only: use executor text as the response
                if instant:
                    task_text = result.text or "Done."
                elif bg_task_ids:
                    task_text = "I'm working on that in the background. I'll let you know when it's ready."
                else:
                    task_text = "Done."
                self.add_message(session_id, "assistant", task_text)
                self.save_chat_session(session_id)
                yield task_text
            return

        # --- Realtime path ---
        if category == CATEGORY_REALTIME and self.realtime_service:
            full_response = ""
            for chunk in self.realtime_service.stream_response(
                question=user_message,
                chat_history=chat_history_pairs
            ):
                if isinstance(chunk, str):
                    full_response += chunk
                yield chunk
            if full_response:
                self.add_message(session_id, "assistant", full_response)
                self.save_chat_session(session_id)
            return

        # --- General (default) path ---
        full_response = ""
        for chunk in self.groq_service.stream_response(
            question=user_message,
            chat_history=chat_history_pairs
        ):
            if isinstance(chunk, str):
                full_response += chunk
            yield chunk

        if full_response:
            self.add_message(session_id, "assistant", full_response)
            self.save_chat_session(session_id)

    # ---------------------------------------------------------------------
    # SAVE SESSION
    # ---------------------------------------------------------------------

    def save_chat_session(
        self,
        session_id: str
    ) -> None:

        messages = self.sessions.get(session_id)

        if not messages:
            return

        filepath = self.get_session_filepath(session_id)

        chat_dict = {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in messages
            ]
        }

        try:

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    chat_dict,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            logger.info(
                "Saved session: %s",
                session_id
            )

        except Exception as e:
            logger.error(
                "Failed to save session %s: %s",
                session_id,
                e
            )