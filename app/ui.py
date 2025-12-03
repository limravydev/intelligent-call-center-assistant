import os
import re
from datetime import datetime
import time
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Load .env
load_dotenv()

# ====================== BACKEND IMPORTS ======================
from app.rag import build_or_load_index
from app.chatbot import init_gemini_client, answer_question

# =========================== PAGE CONFIG ===========================

st.set_page_config(
    page_title="Call Center RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"  # Forces sidebar open so you see the uploader
)

# ============================= CSS ================================

st.markdown(
    """
<style>
    /* 1. GLOBAL FONTS & RESET */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
        background-color: #f3f4f6;
        color: #1f2937;
    }
    
    /* 2. HIDE DEFAULT ELEMENTS */
    header[data-testid="stHeader"] { display: none; }
    footer { display: none; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
    }

    /* 3. HEADER STYLING */
    .header-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .header-icon {
        background: #3b82f6; color: white; width: 36px; height: 36px;
        border-radius: 8px; display: flex; align-items: center; justify-content: center;
        font-size: 20px;
    }
    .header-title-text { font-size: 18px; font-weight: 700; color: #111827; line-height: 1.2; }
    .header-subtitle { font-size: 12px; color: #6b7280; font-weight: 400; }

    /* 4. CHAT MESSAGE STYLES */
    .user-msg-container {
        display: flex; flex-direction: column; align-items: flex-end;
        margin-bottom: 16px; padding-right: 10px;
    }
    .user-bubble {
        background-color: #2563eb; color: white;
        padding: 12px 18px; border-radius: 18px 18px 4px 18px;
        font-size: 15px; max-width: 85%;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    .msg-timestamp { font-size: 10px; color: #9ca3af; margin-top: 4px; margin-right: 4px; }

    .agent-msg-container { margin-bottom: 24px; padding-left: 2px; }
    .agent-label {
        font-size: 11px; font-weight: 600; color: #6b7280;
        margin-bottom: 6px; margin-left: 4px; text-transform: uppercase;
    }
    
    /* SOFT AGENT CARD STYLE */
    .agent-card {
        background: linear-gradient(to bottom, #ffffff, #f7fbff); 
        border: 1px solid #eef2f6;
        border-radius: 16px; 
        padding: 24px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02); 
        width: 100%;
    }

    /* Agent Content */
    .answer-badge {
        display: inline-flex; align-items: center;
        background-color: #dbeafe; color: #1e40af;
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        padding: 4px 10px; border-radius: 99px; margin-bottom: 12px;
    }
    .answer-badge::before {
        content: ""; display: inline-block; width: 6px; height: 6px;
        background-color: #2563eb; border-radius: 50%; margin-right: 6px;
    }
    .answer-text { font-size: 15px; line-height: 1.6; color: #1f2937; margin-bottom: 16px; }

    .notes-box {
        background-color: #fffbeb; border: 1px solid #fcd34d;
        border-radius: 12px; padding: 14px; margin-bottom: 16px;
    }
    .notes-header {
        font-size: 11px; font-weight: 700; color: #92400e;
        text-transform: uppercase; margin-bottom: 4px;
    }
    .notes-content { font-size: 14px; color: #92400e; line-height: 1.4; }

    .steps-header {
        font-size: 11px; font-weight: 700; color: #4b5563;
        text-transform: uppercase; margin-bottom: 8px;
    }
    .steps-list { padding-left: 20px; margin: 0; font-size: 14px; color: #1f2937; }
    .steps-list li { margin-bottom: 4px; }

    /* 5. CONTEXT PANEL (Dark Navy) */
    .context-card {
        background-color: #1e293b; /* Dark Navy */
        border-radius: 20px; padding: 24px;
        color: #f1f5f9;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        height: 100%; min-height: 600px;
        border: 1px solid #334155;
    }
    .context-header {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px;
    }
    .context-label { font-size: 11px; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.1em;}
    
    .doc-snippet-label { font-size: 10px; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; letter-spacing: 0.05em;}
    .doc-title { color: #60a5fa; font-size: 14px; font-weight: 600; margin-bottom: 8px; }
    .doc-text { color: #e2e8f0; font-size: 13px; line-height: 1.6; margin-bottom: 12px; }
    .doc-source { font-size: 11px; color: #64748b; }

    /* 6. INPUT FIX */
    .stTextInput > div > div {
        border-radius: 99px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        background: #ffffff !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    .stTextInput input { color: #1f2937 !important; }

    /* 7. BUTTON STYLING */
    button[kind="primary"] {
        border-radius: 99px !important;
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        height: 48px !important;
    }
    button[kind="primary"]:hover { background-color: #1d4ed8 !important; }

    button[kind="secondary"] {
        background-color: #fef2f2 !important; 
        border: 1px solid #fecaca !important;
        color: #991b1b !important;
        font-weight: 600 !important;
        height: 40px !important;
    }
    button[kind="secondary"]:hover {
        background-color: #fee2e2 !important;
        border-color: #f87171 !important;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

</style>
    """,
    unsafe_allow_html=True,
)

# ====================== BACKEND INITIALIZATION ======================

@st.cache_resource
def load_backend():
    collection, embed_model = build_or_load_index(rebuild=False)
    gemini_client = init_gemini_client()
    return collection, embed_model, gemini_client

collection, embed_model, gemini_client = load_backend()

# ====================== SESSION STATE ======================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False

# ====================== LOGIC & HELPERS ======================

def clear_chat():
    st.session_state.messages = []
    st.session_state.user_query = ""
    st.session_state.processing_query = False

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to the correct directory based on extension."""
    if not uploaded_file: return None
    
    # Check data folder exists
    if not os.path.exists("data"): os.makedirs("data")
    
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Save logic matching your folder structure
    if file_ext == ".pdf":
        save_dir = os.path.join("data", "pdf")
    elif file_ext in [".xlsx", ".xls"]:
        save_dir = os.path.join("data", "excel")
    else:
        st.error(f"Unsupported file type: {file_ext}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path

def split_answer_sections(answer_text: str):
    lower = answer_text.lower()
    ca_idx = lower.find("customer answer:")
    in_idx = lower.find("internal notes:")
    st_idx = lower.find("steps:")

    if ca_idx == -1: return answer_text, None, None

    def slice_part(start, end):
        if start == -1: return None
        part = answer_text[start:] if end == -1 else answer_text[start:end]
        colon_pos = part.find(":")
        if colon_pos != -1: part = part[colon_pos + 1 :]
        return part.strip()

    customer_part = slice_part(ca_idx, in_idx if in_idx != -1 else st_idx)
    notes_part = slice_part(in_idx, st_idx) if in_idx != -1 else None
    steps_part = slice_part(st_idx, -1) if st_idx != -1 else None
    return customer_part, notes_part, steps_part

def steps_to_html_list(steps_text: str) -> str:
    if not steps_text: return ""
    cleaned = re.sub(r"(?i)^\s*steps?:", "", steps_text).strip()
    parts = re.split(r"\s*\d+\.\s+", cleaned)
    items = [p.strip() for p in parts if p.strip()]
    if len(items) <= 1: return f"<li>{cleaned}</li>"
    return "".join(f"<li>{stp}</li>" for stp in items)

def handle_send():
    """Triggered when user clicks Send. Updates UI immediately."""
    query = st.session_state.user_query.strip()
    if not query: return

    # 1. Append User Msg
    st.session_state.messages.append({
        "role": "user", 
        "content": query,
        "time": datetime.now().strftime("%H:%M")
    })
    
    # 2. Clear input
    st.session_state.user_query = ""
    
    # 3. Set flag
    st.session_state.processing_query = True

# --- REAL DATA EXTRACTION ---
def extract_source_info(text: str):
    if not text: return None
    meta = {}
    source_match = re.search(r"(?:Source|File):\s*([^\n•]+)", text, re.IGNORECASE)
    if source_match:
        full_source = source_match.group(1).strip()
        meta["source"] = full_source
        page_match = re.search(r"(?:Row|Page)\s*(\d+)", full_source, re.IGNORECASE)
        if page_match:
            meta["page"] = f"Pg {page_match.group(1)}"
            meta["source"] = re.sub(r"[-–]\s*(Row|Page).*", "", full_source).strip()
    return meta if meta else None

# =========================== SIDEBAR: KNOWLEDGE BASE ===========================

with st.sidebar:
    st.title("🗂️ Knowledge Base")
    st.caption("Upload new policies or FAQs to update the AI brain.")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload files", 
        type=["pdf", "xlsx"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("Process & Update KB", type="primary", use_container_width=True):
            with st.status("Ingesting documents...", expanded=True) as status:
                for file in uploaded_files:
                    st.write(f"📂 Saving {file.name}...")
                    save_uploaded_file(file)
                
                st.write("🧠 Rebuilding RAG Index (this may take a moment)...")
                
                # FORCE REBUILD: passing rebuild=True to your backend function
                # This ensures the new files are read and embedded
                build_or_load_index(rebuild=True) 
                
                status.update(label="Knowledge Base Updated!", state="complete", expanded=False)
            
            st.success("New knowledge added successfully!")
            time.sleep(2)
            st.rerun()

    st.markdown("---")
    # Quick info
    st.info(f"**Backend:** Gemini + ChromaDB\n\n**Last active:** {datetime.now().strftime('%H:%M')}")


# ============================ UI LAYOUT ============================

# --- 1. HEADER ROW ---
h1, h2 = st.columns([8, 1.5], vertical_alignment="center")

with h1:
    st.markdown("""
    <div class="header-container" style="margin-bottom: 0; border: none; box-shadow: none;">
        <div class="header-left">
            <div class="header-icon">☎</div>
            <div>
                <div class="header-title-text">Call Center RAG Assistant</div>
                <div class="header-subtitle">Internal assistant for agents • Excel + PDF knowledge base</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with h2:
    if st.button("End Call / Clear", type="secondary", use_container_width=True, key="clear_btn"):
        clear_chat()
        st.rerun()

st.markdown("<hr style='margin: 10px 0 20px 0; border: 0; border-top: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

# --- 2. MAIN GRID ---
c1, c2 = st.columns([1.8, 1])

# --- LEFT COLUMN: Chat ---
with c1:
    chat_container = st.container(height=600)
    
    with chat_container:
        # Welcome
        if not st.session_state.messages and not st.session_state.processing_query:
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: #9ca3af; margin-top: 50px;">
                <div style="font-size: 40px; margin-bottom: 10px;">👋</div>
                <div style="font-weight: 600; font-size: 16px; color: #4b5563;">Ready for new call</div>
                <div style="font-size: 13px;">Type the customer's question below.</div>
            </div>
            """, unsafe_allow_html=True)
            
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg-container'><div class='user-bubble'>{msg['content']}</div><div class='msg-timestamp'>You • {msg.get('time', '')}</div></div>", unsafe_allow_html=True)
            
            elif msg["role"] == "assistant":
                if not msg.get("content"): continue
                
                # HTML Construction
                html = f"<div class='agent-msg-container'><div class='agent-label'>Agent • {msg.get('time', '')}</div><div class='agent-card'>"
                
                if msg.get("customer_answer"):
                    html += f"<div class='answer-badge'>Customer Answer</div><div class='answer-text'>{msg['customer_answer']}</div>"
                
                if msg.get("internal_notes"):
                    html += f"<div class='notes-box'><div class='notes-header'>Internal Notes</div><div class='notes-content'>{msg['internal_notes']}</div></div>"
                
                if msg.get("steps"):
                    s_list = steps_to_html_list(msg["steps"])
                    html += f"<div class='steps-header'>Steps</div><ul class='steps-list'>{s_list}</ul>"
                
                if not (msg.get("customer_answer") or msg.get("internal_notes") or msg.get("steps")):
                    html += f"<div>{msg['content']}</div>"

                html += "</div></div>"
                st.markdown(html, unsafe_allow_html=True)
        
        # --- PROCESSING INDICATOR ---
        if st.session_state.processing_query:
            with st.status("Thinking...", expanded=True) as status:
                st.write("🔍 Searching Knowledge Base...")
                
                # 1. Build History
                history_pairs = []
                for m in st.session_state.messages:
                    if m["role"] == "user":
                        history_pairs.append(("user", m["content"]))
                    elif m["role"] == "assistant":
                        history_pairs.append(("assistant", m.get("content", "")))
                
                # 2. Call Backend
                query_text = st.session_state.messages[-1]["content"]
                raw_answer, ctx_preview = answer_question(
                    query_text, collection, embed_model, gemini_client, history=history_pairs
                )
                
                st.write("✨ Synthesizing Answer...")
                time.sleep(0.3)
                
                # 3. Parse
                customer_part, notes_part, steps_part = split_answer_sections(raw_answer)
                
                # 4. Save
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": raw_answer,
                    "customer_answer": customer_part,
                    "internal_notes": notes_part,
                    "steps": steps_part,
                    "ctx_preview": ctx_preview,
                    "time": datetime.now().strftime("%H:%M")
                })
                
                status.update(label="Complete!", state="complete", expanded=False)
                st.session_state.processing_query = False
                st.rerun()

        # Marker
        st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)

    # Input Area
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    input_c1, input_c2 = st.columns([0.85, 0.15], vertical_alignment="bottom")
    with input_c1:
        st.text_input("Query", key="user_query", placeholder="Type the customer's question...", label_visibility="collapsed")
    with input_c2:
        st.button("Send ➤", on_click=handle_send, use_container_width=True, type="primary")


# --- RIGHT COLUMN: Context ---
with c2:
    latest_ctx = None
    real_metadata = None

    for m in reversed(st.session_state.messages):
        if m["role"] == "assistant" and m.get("ctx_preview"):
            latest_ctx = m["ctx_preview"]
            # Extract REAL data dynamically
            real_metadata = extract_source_info(latest_ctx)
            break
    
    # Stable container for right side
    context_container = st.container(height=670) 
    with context_container:
        if latest_ctx:
            # FIX: Construct HTML without indentation
            html = "<div class='context-card' style='min-height: 100%; border: none; box-shadow: none;'>"
            
            # Header - CLEAN
            html += "<div class='context-header'><span class='context-label'>Retrieved Context</span></div>"
            
            # Badges (Only render if metadata was actually found)
            if real_metadata:
                html += "<div style='display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;'>"
                if 'source' in real_metadata:
                    html += f"<div style='background: #334155; color: #e2e8f0; font-size: 10px; padding: 4px 8px; border-radius: 6px; font-weight: 600; display: flex; align-items: center; gap: 4px;'>📄 {real_metadata['source']}</div>"
                if 'page' in real_metadata:
                    html += f"<div style='background: #334155; color: #94a3b8; font-size: 10px; padding: 4px 8px; border-radius: 6px;'>📍 {real_metadata['page']}</div>"
                html += "</div>"

            # Content
            html += "<div class='doc-snippet-label'>Document Excerpt</div>"
            html += f"<div class='doc-text' style='border-left: 3px solid #60a5fa; padding-left: 12px; margin-left: 2px;'>{latest_ctx}</div>"
            
            # Footer removed entirely
            html += "</div>"

            st.markdown(html, unsafe_allow_html=True)
        else:
            # Fallback Empty State
            st.markdown("""
            <div class='context-card' style='min-height: 100%; border: none; box-shadow: none; display: flex; align-items: center; justify-content: center; opacity: 0.8;'>
                <div style='text-align: center; color: #94a3b8; font-size: 13px;'>
                    <div>No context active</div>
                    <div style='font-size: 11px; margin-top: 4px;'>Waiting for query...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================== STICKY AUTO-SCROLL JS ====================
run_key = f"{len(st.session_state.messages)}_{st.session_state.processing_query}"
components.html(
    f"""
    <script>
        scrollBottom();
        let checkCount = 0;
        const intervalId = setInterval(function() {{
            scrollBottom();
            checkCount++;
            if (checkCount > 20) {{ clearInterval(intervalId); }}
        }}, 100);

        function scrollBottom() {{
            const bottomMarker = window.parent.document.getElementById('chat-bottom');
            if (bottomMarker) {{
                bottomMarker.scrollIntoView({{behavior: "smooth", block: "end"}});
            }}
        }}
    </script>
    <div style="display:none;">{run_key}</div>
    """,
    height=0, width=0
)