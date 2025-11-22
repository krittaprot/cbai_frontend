import base64
import json
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from typing import cast

load_dotenv(override=True)

# --------------------------------------
# Configuration
# --------------------------------------
st.set_page_config(page_title="Crystal Blood AI 🩸", page_icon="🤖", layout="centered")
RECENT_TURNS_LIMIT = 3
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")  # 8-digit static password
SESSION_MAX_AGE_SECONDS = 30 * 60  # 30 minutes

# AgentCore runtime configuration (Auth0 + endpoint)
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AUDIENCE = os.getenv("AUDIENCE")
REGION_NAME = os.getenv("REGION_NAME", "ap-southeast-1")
INVOKE_AGENT_ARN = os.getenv("INVOKE_AGENT_ARN")
QUALIFIER = os.getenv("QUALIFIER", "DEFAULT") or "DEFAULT"
AGENTCORE_TRACE_ID = os.getenv("AGENTCORE_TRACE_ID", "trace-ui-streamlit")


def _compose_agentcore_url() -> Optional[str]:
    if not INVOKE_AGENT_ARN:
        return None
    escaped_arn = quote(INVOKE_AGENT_ARN, safe="")
    qualifier = quote(QUALIFIER, safe="")
    return (
        f"https://bedrock-agentcore.{REGION_NAME}.amazonaws.com/runtimes/{escaped_arn}/invocations"
        f"?qualifier={qualifier}"
    )


AGENTCORE_BACKEND_URL = _compose_agentcore_url()
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", AGENTCORE_BACKEND_URL or "http://localhost:8080/invocations")
_JWT_CACHE: Dict[str, Any] = {"token": None, "expires_at": 0.0}

# --------------------------------------
# Data models
# --------------------------------------
@dataclass
class ImagePayload:
    bytes_data: bytes
    mime_type: str
    name: str


# --------------------------------------
# MIME helpers (align with test_agent.py)
# --------------------------------------
MIME_FIX = {
    "image/x-png": "image/png",
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
}


def normalize_mime_type(mime: Optional[str]) -> str:
    if not mime:
        return "image/png"
    mime_lower = mime.lower()
    return MIME_FIX.get(mime_lower, mime_lower)


# --------------------------------------
# AgentCore helpers
# --------------------------------------
def _is_agentcore_endpoint(url: Optional[str]) -> bool:
    if not url:
        return False
    return "bedrock-agentcore." in url


def _agentcore_config_missing() -> List[str]:
    mapping = {
        "AUTH0_DOMAIN": AUTH0_DOMAIN,
        "CLIENT_ID": CLIENT_ID,
        "CLIENT_SECRET": CLIENT_SECRET,
        "AUDIENCE": AUDIENCE,
        "INVOKE_AGENT_ARN": INVOKE_AGENT_ARN,
    }
    return [name for name, value in mapping.items() if not value]


def _get_jwt_token(force: bool = False) -> str:
    cached_token = _JWT_CACHE.get("token")
    expires_at = float(_JWT_CACHE.get("expires_at", 0.0) or 0.0)
    now = time.time()
    if not force and cached_token and now < (expires_at - 30):
        return cast(str, cached_token)

    missing = _agentcore_config_missing()
    if missing:
        raise RuntimeError(
            "Missing AgentCore configuration: " + ", ".join(missing)
        )

    token_url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "audience": AUDIENCE,
    }

    response = requests.post(token_url, json=payload, headers={"content-type": "application/json"}, timeout=15)
    response.raise_for_status()
    data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Auth0 token response missing access_token")

    expires_in = float(data.get("expires_in", 300) or 300)
    _JWT_CACHE["token"] = token
    _JWT_CACHE["expires_at"] = now + max(60.0, expires_in)
    return token


def _agentcore_headers(session_id: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_jwt_token()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id or str(uuid.uuid4()),
        "X-Amzn-Trace-Id": AGENTCORE_TRACE_ID,
    }


def invoke_agentcore_endpoint(url: str, payload: Dict[str, Any], session_id: str) -> requests.Response:
    body = json.dumps(payload)
    headers = _agentcore_headers(session_id)
    return requests.post(
        url,
        headers=headers,
        data=body,
        stream=True,
        timeout=(10, 300),
    )


# --------------------------------------
# Session state initialization
# --------------------------------------
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("conversation_id", str(uuid.uuid4()))
    ss.setdefault("messages", [])
    ss.setdefault("backend_url", DEFAULT_BACKEND_URL)
    ss.setdefault("backend_status", "unknown")
    ss.setdefault("backend_latency_ms", None)
    ss.setdefault("recent_turns", [])
    ss.setdefault("active_turn", None)
    # Auth-related session keys
    ss.setdefault("auth_ok", False)
    ss.setdefault("auth_time", None)


# --------------------------------------
# Simple password auth (8 digits, 30 min)
# --------------------------------------
def _auth_session_valid() -> bool:
    ss = st.session_state
    if not ss.get("auth_ok"):
        return False
    ts = ss.get("auth_time")
    if not isinstance(ts, (int, float)):
        ss.auth_ok = False
        ss.auth_time = None
        return False
    # Fixed 30-minute window from login time
    if (time.time() - float(ts)) <= SESSION_MAX_AGE_SECONDS:
        return True
    # Expired
    ss.auth_ok = False
    ss.auth_time = None
    return False

# --------------------------------------
# Improved login UI
# --------------------------------------
def _render_login_gate() -> None:
    st.markdown(
        """
        <style>
        /* Hide default Streamlit elements on login page */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 70vh;
        }
        .login-card {
            max-width: 420px;
            width: 100%;
            padding: 48px 40px;
            border-radius: 16px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            border: 1px solid rgba(0,0,0,0.05);
        }
        .login-icon {
            text-align: center;
            font-size: 64px;
            margin-bottom: 24px;
        }
        .login-title {
            text-align: center;
            font-size: 32px;
            margin-bottom: 8px;
            font-weight: 700;
            color: #1a1a1a;
            letter-spacing: -0.5px;
        }
        .login-caption {
            text-align: center;
            color: #6c757d;
            margin-bottom: 36px;
            font-size: 15px;
            line-height: 1.5;
        }
        .stTextInput > div > div > input {
            text-align: center;
            font-size: 18px;
            letter-spacing: 4px;
            font-weight: 500;
            padding: 14px;
            border-radius: 10px;
        }
        .stButton > button {
            width: 100%;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Center the login card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-icon">🩸</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Crystal Blood AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-caption">Enter your 8-digit access code to continue</div>',
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            pwd = st.text_input(
                "Access Code",
                type="password",
                max_chars=8,
                placeholder="••••••••",
                label_visibility="collapsed"
            ).strip()

            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if pwd == AUTH_PASSWORD and len(pwd) == 8 and pwd.isdigit():
                    st.session_state.auth_ok = True
                    st.session_state.auth_time = time.time()
                    st.success("✓ Login successful")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("✗ Invalid access code. Please try again.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align: center; color: #adb5bd; font-size: 13px;">Veterinary AI Assistant • Secure Access</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------
# Input parsing
# --------------------------------------
def parse_chat_input_value(value: Any) -> Tuple[str, List[Any]]:
    """Normalize st.chat_input return value (text + optional files)."""
    if value is None:
        return "", []
    if isinstance(value, str):
        return value, []

    text_fields = ("text", "message", "prompt", "value")
    files_fields = ("files", "file", "uploaded_files", "uploads")

    def extract(getter) -> Tuple[str, List[Any]]:
        text = ""
        files: List[Any] = []
        for f in text_fields:
            v = getter(f)
            if isinstance(v, str):
                text = v
                break
        for f in files_fields:
            v = getter(f)
            if not v:
                continue
            files = list(v) if isinstance(v, (list, tuple, set)) else [v]
            break
        return text, files

    if isinstance(value, dict):
        return extract(value.get)
    return extract(lambda k: getattr(value, k, None))


# --------------------------------------
# Image handling
# --------------------------------------
def uploaded_files_to_payloads(files: Sequence[Any]) -> List[ImagePayload]:
    """Convert uploaded files to ImagePayload objects."""
    payloads: List[ImagePayload] = []
    for idx, f in enumerate(files):
        name = getattr(f, "name", None) or f"upload-{idx + 1}"
        mime = getattr(f, "type", None)
        data: Optional[bytes] = None

        try:
            getvalue = getattr(f, "getvalue", None)
            read = getattr(f, "read", None)
            if callable(getvalue):
                data = getvalue()
            elif callable(read):
                data = read()
                seek = getattr(f, "seek", None)
                if callable(seek):
                    try:
                        seek(0)
                    except Exception:
                        pass
        except Exception:
            data = None

        if not data:
            continue

        mime = mime or mimetypes.guess_type(name)[0] or "application/octet-stream"
        payloads.append(ImagePayload(bytes_data=data, mime_type=mime, name=name))
    return payloads


def encode_image_payload(payload: ImagePayload) -> Dict[str, str]:
    """Encode image payload for API request."""
    return {
        "base64": base64.b64encode(payload.bytes_data).decode("utf-8"),
        "mime_type": normalize_mime_type(payload.mime_type),
    }


def serialize_image_for_memory(payload: ImagePayload) -> Dict[str, str]:
    """Serialize image for session state storage."""
    return {
        "base64": base64.b64encode(payload.bytes_data).decode("utf-8"),
        "mime_type": payload.mime_type,
        "name": payload.name,
    }


def serialize_image_info_for_memory(image_info: Dict) -> Dict[str, str]:
    """Serialize image info for session state storage."""
    if "bytes" in image_info:
        return {
            "base64": base64.b64encode(image_info["bytes"]).decode("utf-8"),
            "mime_type": image_info.get("mime", "image/png"),
            "name": image_info.get("name"),
        }
    if "url" in image_info:
        return {"url": image_info["url"], "name": image_info.get("name")}
    return {}


def build_agent_request_payload(message_text: str, images: Sequence[ImagePayload]) -> Dict[str, Any]:
    """Build payload compatible with AgentCore invocation (matches test_agent.py)."""
    cleaned_text = (message_text or "").strip()
    encoded_images = [encode_image_payload(img) for img in images]

    if not cleaned_text and not encoded_images:
        return {}

    input_payload: Dict[str, Any] = {}
    if cleaned_text:
        input_payload["text"] = cleaned_text
    if encoded_images:
        if cleaned_text:
            # Provide prompt fallback for backward compatibility (same as test_agent.py).
            input_payload["prompt"] = cleaned_text
        input_payload["images"] = encoded_images

    return {"input": input_payload}


def image_part_from_response(part: Dict) -> Optional[Dict]:
    """Extract image data from API response part."""
    if "image_base64" in part and isinstance(part["image_base64"], str):
        try:
            raw = part["image_base64"]
            b64 = raw.split(",", 1)[1] if raw.startswith("data:") else raw
            image_bytes = base64.b64decode(b64)
            return {
                "bytes": image_bytes,
                "mime": part.get("mime_type", "image/png"),
                "name": part.get("name"),
            }
        except Exception:
            return None

    url = part.get("url") or part.get("image_url")
    if isinstance(url, str):
        return {"url": url, "name": part.get("name")}
    return None


def decode_stream_chunk(raw_chunk: str) -> Optional[Any]:
    """Decode SSE payloads that may contain nested JSON strings."""
    try:
        payload: Any = json.loads(raw_chunk)
    except json.JSONDecodeError:
        return None

    while isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            break

    return payload


def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single Server-Sent Event line into a dict."""
    if not line:
        return None
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if stripped.startswith("data:"):
        stripped = stripped[len("data:") :].strip()
    if not stripped:
        return None
    payload = decode_stream_chunk(stripped)
    return payload if isinstance(payload, dict) else None


# --------------------------------------
# Download helpers
# --------------------------------------
def extract_text_from_parts(parts: Sequence[Dict[str, Any]]) -> str:
    """Collect the textual segments from message parts."""
    texts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        value = part.get("value")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                texts.append(stripped)
    return "\n\n".join(texts).strip()


def render_download_controls(text: str, filename_prefix: str, key_suffix: str) -> None:
    """Render a stylized download control for the plain-text export."""
    safe_text = (text or "").strip()
    if not safe_text:
        return
    safe_prefix = (filename_prefix or "cbai-response").strip().replace(" ", "-") or "cbai-response"
    txt_bytes = safe_text.encode("utf-8")
    encoded = base64.b64encode(txt_bytes).decode("utf-8")
    container_id = f"cbai-download-{key_suffix}"
    button_html = f"""
        <style>
            #{container_id} {{
                margin: 0.5rem 0 1rem 0;
                padding: 1rem;
                border-radius: 14px;
                border: 1px solid rgba(99, 102, 241, 0.3);
                background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
            }}
            #{container_id} .download-title {{
                font-size: 0.85rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #5a4ab8;
                margin-bottom: 0.35rem;
            }}
            #{container_id} .download-btn {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.6rem 1.5rem;
                border-radius: 999px;
                border: none;
                background: linear-gradient(120deg, #6f6cf6, #a855f7);
                color: white;
                font-weight: 600;
                font-size: 0.95rem;
                text-decoration: none;
                box-shadow: 0 8px 20px rgba(111, 108, 246, 0.35);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }}
            #{container_id} .download-btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 12px 24px rgba(111, 108, 246, 0.45);
            }}
        </style>
        <div id="{container_id}">
            <div class="download-title">Save this AI summary</div>
            <a class="download-btn" download="{safe_prefix}.txt" href="data:text/plain;base64,{encoded}">⬇️ Download as TXT</a>
        </div>
    """
    st.markdown(button_html, unsafe_allow_html=True)


# --------------------------------------
# Message rendering
# --------------------------------------
def avatar_for_role(role: str) -> str:
    """Return emoji avatar for message role."""
    return "🤔" if role == "user" else "👨🏻‍⚕️"


def render_message(message: Dict, show_images: bool = False, index: Optional[int] = None) -> None:
    """Render a chat message with its parts."""
    role = message.get("role", "assistant")
    parts: List[Dict] = message.get("parts", [])
    download_text = extract_text_from_parts(parts) if role == "assistant" else ""

    with st.chat_message(role, avatar=avatar_for_role(role)):
        for part in parts:
            ptype = part.get("type")
            if ptype == "text" and isinstance(part.get("value"), str):
                st.markdown(part["value"])
            elif ptype == "image" and show_images:
                data = part.get("value") or {}
                if "bytes" in data:
                    st.image(data["bytes"], caption=data.get("name"), use_container_width=True)
                elif "url" in data:
                    st.image(data["url"], caption=data.get("name"), use_container_width=True)
            elif ptype == "error":
                st.error(part.get("value", "An unknown error occurred."))
        if role == "assistant" and download_text:
            suffix = str(index) if index is not None else f"latest_{id(message)}"
            prefix = (
                f"cbai-response-{index + 1}" if index is not None else "cbai-response-latest"
            )
            render_download_controls(download_text, prefix, suffix)


# --------------------------------------
# Backend health check
# --------------------------------------
def _health_url_from_backend(target: str) -> Optional[str]:
    """Derive a health-check URL based on the configured backend endpoint."""
    if not target:
        return None
    trimmed = target.strip()
    if not trimmed:
        return None
    parsed = urlparse(trimmed)
    if parsed.scheme and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return f"{base}/health"
    trimmed = trimmed.rstrip("/")
    return f"{trimmed}/health" if trimmed else None


def check_backend_health(base_url: str, timeout: float = 3.0) -> Tuple[str, Optional[int]]:
    """Check backend health. Returns (status, latency_ms)."""
    if _is_agentcore_endpoint(base_url):
        return "agentcore", None
    health_url = _health_url_from_backend(base_url or "")
    if not health_url:
        return "unknown", None
    try:
        start = time.perf_counter()
        resp = requests.get(health_url, timeout=timeout)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
    except requests.RequestException:
        return "error", None
    if resp.status_code != 200:
        return "error", elapsed_ms
    try:
        payload = resp.json()
        status_text = str(payload.get("status", "")).lower()
        if status_text in {"ok", "healthy", "up"}:
            return "ok", elapsed_ms
    except ValueError:
        pass
    return "ok", elapsed_ms


def invoke_backend(backend_url: str, payload: Dict[str, Any], session_id: str) -> requests.Response:
    """Invoke either the legacy backend or the deployed AgentCore runtime."""
    if not backend_url:
        raise RuntimeError("Backend URL is not configured.")

    if _is_agentcore_endpoint(backend_url):
        missing = _agentcore_config_missing()
        if missing:
            raise RuntimeError("Missing AgentCore configuration: " + ", ".join(missing))
        return invoke_agentcore_endpoint(backend_url, payload, session_id)

    return requests.post(
        backend_url,
        json=payload,
        stream=True,
        timeout=(5, 180),
    )


def render_backend_status() -> None:
    """Render compact backend status indicator."""
    st.sidebar.divider()
    st.sidebar.subheader("Server Status")
    backend_url = (st.session_state.backend_url or "").strip().rstrip("/")
    st.session_state.backend_url = backend_url

    is_agentcore = _is_agentcore_endpoint(backend_url)
    status = "ok" if is_agentcore else st.session_state.backend_status
    is_online = status != "error"

    if is_online:
        st.sidebar.success("🟢 Online", icon="✅")
    else:
        st.sidebar.error("🔴 Offline", icon="❌")

    if not is_agentcore and st.sidebar.button("Refresh Status", use_container_width=True):
        with st.spinner(""):
            status, latency = check_backend_health(st.session_state.backend_url)
            st.session_state.backend_status = status
            st.session_state.backend_latency_ms = latency
            st.rerun()


# --------------------------------------
# Sidebar controls
# --------------------------------------
def sidebar_controls() -> None:
    """Render sidebar controls."""
    st.sidebar.title("Control Panel")
    # Optional: show session status and a logout button
    if _auth_session_valid():
        remaining = int(max(0, SESSION_MAX_AGE_SECONDS - (time.time() - cast(float, st.session_state.auth_time or 0))))
        mins = remaining // 60
        # st.sidebar.info(f"Session: {mins} min left", icon="⏳")
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state.auth_ok = False
            st.session_state.auth_time = None
            st.rerun()
    if st.sidebar.button("💬 New Conversation", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.recent_turns = []
        st.session_state.active_turn = None
        st.rerun()
    render_backend_status()


# --------------------------------------
# Main application
# --------------------------------------
def main() -> None:
    init_state()
    # Gate the entire app behind a simple password login
    if not _auth_session_valid():
        _render_login_gate()
        return
    sidebar_controls()

    st.title("Crystal Blood AI 🩸")
    st.caption("Upload pet blood test reports and get veterinary AI insights.")

    # create a single placeholder for the center tips so we can clear it immediately
    tips_placeholder = st.empty()

    # Center info only when no messages
    if not st.session_state.messages:
        with tips_placeholder.container():
            st.markdown(
                """
                <style>
                    .tips-box {
                        border-radius: 12px;
                        padding: 1.25rem;
                        margin-bottom: 1rem;
                    }

                    .tips-box.primary {
                        border: 1px solid rgba(222, 182, 255, 0.8);
                        background: rgba(241, 228, 255, 0.65);
                    }

                    .tips-box.secondary {
                        border: 1px solid rgba(255, 205, 178, 0.9);
                        background: rgba(255, 239, 229, 0.75);
                    }

                    .tips-title {
                        margin-top: 0;
                        margin-bottom: 0.75rem;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="tips-box primary">
                    <h4 class="tips-title">🚀 Quickstart</h4>
                    <ul>
                        <li>Upload CBC, biochemistry, and/or blood gas reports (PNG, JPG, JPEG). Send one, two, or all three together.</li>
                        <li>Crystal Blood AI reads every panel, flags abnormal patterns, and prepares an initial explanation.</li>
                        <li>After you review the AI output, ask specific follow-up questions to dive deeper into the highlighted values.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="tips-box secondary">
                    <h4 class="tips-title">💡 Helpful Tips</h4>
                    <ul>
                        <li>Use crisp, full-page images so every result row is legible.</li>
                        <li>Describe your pet's history or symptoms in plain language&mdash;medical jargon is optional.</li>
                        <li>Only the last 3 question/answer turns are kept for context, so restate key info if a conversation runs long.</li>
                        <li>Try follow-ups like <em>"Why did the AI flag the potassium value?"</em> or <em>"What should I monitor after these CBC results?"</em></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Crystal Blood AI helps veterinarians and pet parents interpret lab work with clear, actionable explanations for cats and dogs."
            )

    # Render chat history
    for idx, message in enumerate(st.session_state.messages):
        render_message(message, show_images=True, index=idx)

    # Chat input
    chat_value = st.chat_input(
        "Ask about your pet's blood test results...",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg"],
    )
    if chat_value is None:
        return

    # Parse input
    raw_text, uploaded_files = parse_chat_input_value(chat_value)
    message_text = (raw_text or "").strip()
    image_payloads = uploaded_files_to_payloads(uploaded_files)
    if uploaded_files and not image_payloads:
        st.warning("⚠️ Could not process attached files. Please use supported image formats (PNG, JPG, JPEG).")
        return
    if not (message_text or image_payloads):
        return

    # Build user message
    user_parts: List[Dict] = []
    if message_text:
        user_parts.append({"type": "text", "value": message_text})
    for p in image_payloads:
        user_parts.append({"type": "image", "value": {"bytes": p.bytes_data, "mime": p.mime_type, "name": p.name}})

    st.session_state.active_turn = {
        "user": {"text": message_text or "", "images": [serialize_image_for_memory(p) for p in image_payloads]},
        "assistant": None,
    }

    user_message = {"role": "user", "parts": user_parts}

    # append the message and clear the tips placeholder immediately so tips disappear in this same run
    st.session_state.messages.append(user_message)
    try:
        tips_placeholder.empty()
    except Exception:
        # fallback: ignore clearing errors so app does not break
        pass

    render_message(user_message, show_images=True, index=len(st.session_state.messages) - 1)

    # Prepare API payload (AgentCore-compatible)
    payload = build_agent_request_payload(message_text, image_payloads)
    if not payload:
        st.warning("⚠️ Unable to build a request payload. Please add text or an image.")
        st.session_state.active_turn = None
        return
    payload["conversation_id"] = st.session_state.conversation_id

    # Stream response
    with st.chat_message("assistant", avatar=avatar_for_role("assistant")):
        text_placeholder = st.empty()
        image_container = st.container()
        assistant_text = ""
        assistant_parts: List[Dict] = []

        backend_url = st.session_state.backend_url
        session_id = st.session_state.conversation_id

        try:
            resp = invoke_backend(backend_url, payload, session_id)
        except RuntimeError as exc:
            msg = f"❌ Configuration error: {exc}"
            text_placeholder.error(msg)
            assistant_parts.append({"type": "error", "value": msg})
            st.session_state.messages.append({"role": "assistant", "parts": assistant_parts})
            st.session_state.active_turn = None
            return
        except requests.RequestException as exc:
            msg = f"❌ Request failed: {exc}"
            text_placeholder.error(msg)
            assistant_parts.append({"type": "error", "value": msg})
            st.session_state.messages.append({"role": "assistant", "parts": assistant_parts})
            st.session_state.active_turn = None
            return

        if resp.status_code != 200:
            msg = f"❌ Backend returned {resp.status_code}: {resp.text}"
            text_placeholder.error(msg)
            assistant_parts.append({"type": "error", "value": msg})
            st.session_state.messages.append({"role": "assistant", "parts": assistant_parts})
            st.session_state.active_turn = None
            return

        final_payload: Any = {}
        content_type = (resp.headers.get("Content-Type") or "").lower()

        def extract_text_from_payload(data: Any) -> str:
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                text_val = data.get("text") or data.get("message")
                if isinstance(text_val, str):
                    return text_val
            return ""

        if "text/event-stream" not in content_type:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            extracted = extract_text_from_payload(body)
            assistant_text_local = extracted or (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False))
            assistant_text_local = (assistant_text_local or "").strip()
            if assistant_text_local:
                text_placeholder.markdown(assistant_text_local)
                assistant_parts.append({"type": "text", "value": assistant_text_local})
            else:
                msg = "❌ Unexpected non-stream response."
                text_placeholder.error(msg)
                assistant_parts.append({"type": "error", "value": msg})
            st.session_state.messages.append({"role": "assistant", "parts": assistant_parts})
            st.session_state.active_turn = None
            return

        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            event = parse_sse_line(raw_line)
            if not event:
                continue

            etype = event.get("type")
            if etype == "delta":
                token = event.get("data", "")
                if isinstance(token, str):
                    assistant_text += token
                    text_placeholder.markdown(assistant_text + "▌")
            elif etype in {"message", "final"}:
                final_payload = event.get("data") or final_payload
                extracted = extract_text_from_payload(final_payload)
                if extracted and not assistant_text:
                    assistant_text = extracted
                    text_placeholder.markdown(assistant_text + ("▌" if etype == "message" else ""))
            elif etype == "error":
                msg = event.get("error") or event.get("data") or "Unknown error"
                text_placeholder.error(msg)
                assistant_parts.append({"type": "error", "value": msg})
                st.session_state.messages.append({"role": "assistant", "parts": assistant_parts})
                st.session_state.active_turn = None
                return
            elif etype == "done":
                break

        assistant_text = assistant_text.strip()
        if assistant_text:
            text_placeholder.markdown(assistant_text)
            assistant_parts.append({"type": "text", "value": assistant_text})
        else:
            text_placeholder.empty()

        pending_index = len(st.session_state.messages)
        render_download_controls(
            assistant_text,
            f"cbai-response-{pending_index + 1}",
            f"live_{st.session_state.conversation_id}_{pending_index}",
        )

        # Show any images in response
        content_items = []
        if isinstance(final_payload, dict):
            content_items = final_payload.get("content") or []
        assistant_memory_images: List[Dict] = []
        for content in content_items:
            image_info = image_part_from_response(content)
            if not image_info:
                continue
            if "bytes" in image_info:
                image_container.image(image_info["bytes"], caption=image_info.get("name"), use_container_width=True)
            elif "url" in image_info:
                image_container.image(image_info["url"], caption=image_info.get("name"), use_container_width=True)
            assistant_parts.append({"type": "image", "value": image_info})
            mem = serialize_image_info_for_memory(image_info)
            if mem:
                assistant_memory_images.append(mem)

        st.session_state.messages.append({"role": "assistant", "parts": assistant_parts})
        current_turn = st.session_state.get("active_turn")
        if current_turn is not None:
            current_turn["assistant"] = {"text": assistant_text, "images": assistant_memory_images}
            st.session_state.recent_turns.append(current_turn)
            if len(st.session_state.recent_turns) > RECENT_TURNS_LIMIT:
                st.session_state.recent_turns = st.session_state.recent_turns[-RECENT_TURNS_LIMIT:]
            st.session_state.active_turn = None


if __name__ == "__main__":
    main()
