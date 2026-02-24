import streamlit as st
import time
import random

# --- Page Config ---
st.set_page_config(
    page_title="Perplexity Clone",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for UI Styling ---
st.markdown("""
<style>
    /* Styling for the Source Cards */
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e0e0e0;
    }
    /* Dark mode adjustment for cards (optional hack) */
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #41444b;
        }
    }
    /* Hide the top decoration */
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# --- Helper: Mock Search & Reasoning ---
def search_and_generate(query):
    """
    Simulates the backend process:
    1. Searching the web
    2. Finding sources
    3. Generating an answer
    """
    
    # 1. Searching Animation
    with st.status("🔍 Searching the web...", expanded=True) as status:
        st.write("Searching for keywords...")
        time.sleep(1)
        st.write("Found 4 relevant sources...")
        time.sleep(0.5)
        st.write("Reading content...")
        time.sleep(0.5)
        status.update(label="✅ Research Complete", state="complete", expanded=False)

    # 2. Mock Sources Data
    sources = [
        {"title": "Wikipedia - Topic", "url": "https://en.wikipedia.org", "snippet": "Detailed info..."},
        {"title": "News Outlet A", "url": "https://news.com", "snippet": "Recent updates..."},
        {"title": "Official Docs", "url": "https://docs.io", "snippet": "Technical specs..."},
        {"title": "Reddit Thread", "url": "https://reddit.com", "snippet": "User discussions..."}
    ]

    # 3. Display Sources (Perplexity Style Grid)
    st.markdown("### Sources")
    cols = st.columns(4)
    for i, source in enumerate(sources):
        with cols[i]:
            st.info(f"**{source['title']}**\n\n[Link]({source['url']})", icon="📄")

    # 4. Stream the Answer
    st.markdown("### Answer")
    response_placeholder = st.empty()
    full_response = f"Here is a generated answer for **'{query}'**. \n\nThis is where the LLM's reasoning would appear. In a real app, you would connect this to the Perplexity API or OpenAI API with search tool invocation. \n\nKey points found:\n- Fact A from Source 1\n- Fact B from Source 2\n\nCitations would be handled here [1]."
    
    # Simulate streaming
    streamed_text = ""
    for char in full_response:
        streamed_text += char
        response_placeholder.markdown(streamed_text + "▌")
        time.sleep(0.015)
    response_placeholder.markdown(streamed_text)

    # Save interaction to history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})


# --- UI Layout Logic ---

# Sidebar for History
with st.sidebar:
    st.title("Library")
    if st.button("New Thread", type="primary"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**History**")
    # Show last 5 queries as history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.caption(f"🔍 {msg['content'][:30]}...")

# Main Content Area
if not st.session_state.messages:
    # --- Home Screen Mode ---
    st.markdown("<br><br><br><br>", unsafe_allow_html=True) # Spacing
    st.markdown("<h1 style='text-align: center;'>Where knowledge begins</h1>", unsafe_allow_html=True)
    
    # Centered search input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        query = st.chat_input("Ask anything...")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()
            
    # Suggestions
    st.markdown("<br>", unsafe_allow_html=True)
    scol1, scol2, scol3 = st.columns([1, 2, 1])
    with scol2:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        if cols[0].button("🚀 latest AI news"):
            st.session_state.messages.append({"role": "user", "content": "latest AI news"})
            st.rerun()
        if cols[1].button("💰 AAPL analysis"):
            st.session_state.messages.append({"role": "user", "content": "AAPL analysis"})
            st.rerun()
        if cols[2].button("🎨 best color schemes"):
            st.session_state.messages.append({"role": "user", "content": "best color schemes"})
            st.rerun()

else:
    # --- Chat Interface Mode ---
    
    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # If it's an assistant message, we might have sources to show
                if "sources" in message:
                    st.markdown("### Sources")
                    cols = st.columns(4)
                    for i, source in enumerate(message["sources"]):
                        with cols[i]:
                            st.info(f"**{source['title']}**", icon="📄")
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # If the last message was from the user, generate a response
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            search_and_generate(st.session_state.messages[-1]["content"])

    # Sticky Bottom Input
    query = st.chat_input("Ask follow-up...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()
