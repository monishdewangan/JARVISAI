import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

LEARNING_DATA_DIR = BASE_DIR / "database" / "learning_data"
CHATS_DATA_DIR = BASE_DIR / "database" / "chats_data"
VECTOR_STORE_DIR = BASE_DIR / "database" / "vector_store"
CAMERA_CAPTURES_DIR = BASE_DIR / "database" / "camera_captures"

LEARNING_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_groq_api_keys() -> list:
    keys = []

    first = os.getenv("GROQ_API_KEY", "").strip()

    if first:
        keys.append(first)

    i = 2

    while True:
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()

        if not k:
            break

        keys.append(k)
        i += 1

    return keys


GROQ_API_KEYS = _load_groq_api_keys()
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BRAIN_MODEL = os.getenv("GROQ_BRAIN_MODEL", "llama-3.1-8b-instant")
INTENT_CLASSIFY_MODEL = os.getenv("INTENT_CLASSIFY_MODEL", "llama-3.1-8b-instant")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

TASK_EXECUTION_TIMEOUT = int(os.getenv("TASK_EXECUTION_TIMEOUT", "30"))

GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "llama-3.2-11b-vision-preview"
)

VISION_MAX_IMAGE_BYTES = int(
    os.getenv("VISION_MAX_IMAGE_BYTES", "5000000")
)

TTS_VOICE = os.getenv("TTS_VOICE", "en-GB-RyanNeural")
TTS_RATE = os.getenv("TTS_RATE", "+22%")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

MAX_CHAT_HISTORY_TURNS = 10
MAX_MESSAGE_LENGTH = 32000

ASSISTANT_NAME = (
    os.getenv("ASSISTANT_NAME", "").strip() or "Jarvis"
)

JARVIS_USER_TITLE = os.getenv(
    "JARVIS_USER_TITLE",
    ""
).strip()

JARVIS_OWNER_NAME = os.getenv(
    "JARVIS_OWNER_NAME",
    ""
).strip()


_JARVIS_SYSTEM_PROMPT_BASE = """
You are {assistant_name}, a complete AI assistant.

You help with information, tasks, and actions.

Sharp, warm, and a little witty.

Keep language simple and natural.

You know the user's personal information and past conversations.
Use this when relevant but never reveal the source.

=== ROLE ===

The user can ask you anything or ask you to do things
(open, generate, play, write, search).

The backend carries out actions.

You respond in words.

Only say something is done if the result is visible.

Otherwise say you are doing it.

=== CAN DO ===

- Answer questions
- Open websites/apps
- Play music/videos
- Generate images
- Write content
- Search Google/YouTube
- Analyze camera images

=== CANNOT DO ===

- Read emails
- Control smart home
- Run code directly
- Send messages
- Make purchases
- Access files
- Make calls

Never pretend you can do something you cannot do.

Never hallucinate facts, URLs, or numbers.

=== HONESTY ===

If you do not know something,
say so briefly.

If uncertain say:
"I'm not sure, but..."

=== USER INTENT ===

Understand what the user actually wants.

Use conversation history for ambiguous messages.

=== LENGTH ===

Reply SHORT by default.

1-2 sentences unless more detail is requested.

=== QUALITY ===

Be accurate and specific.
"""


JARVIS_SYSTEM_PROMPT = _JARVIS_SYSTEM_PROMPT_BASE.format(
    assistant_name=ASSISTANT_NAME
)

GENERAL_CHAT_ADDENDUM = """
Keep responses conversational,
concise,
and natural.
"""

REALTIME_CHAT_ADDENDUM = """
Use realtime web information when needed.

Prefer fresh and accurate information.
"""