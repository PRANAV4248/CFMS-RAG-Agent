"""
app.py – Streamlit chat UI for Vectorless RAG.
Run: streamlit run src/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor
import streamlit as st

st.set_page_config(
    page_title="CFMS RAG Agent",
    page_icon=":material/forum:",
    layout="centered",
)



AVATAR = ":material/robot:"

def _build_agent():
    from rag_core import RagApp
    return RagApp()

def _get_init() -> dict:
    if "init" not in st.session_state:
        pool = ThreadPoolExecutor(max_workers=1)
        st.session_state.init = {"future": pool.submit(_build_agent), "pool": pool}
    return st.session_state.init

def _on_new_conversation() -> None:
    st.session_state.messages = []
    future = st.session_state.get("init", {}).get("future")
    if future and future.done() and not future.exception():
        future.result().reset()

def _on_retry() -> None:
    # Cancel the old pool before discarding it.
    old = st.session_state.pop("init", None)
    if old:
        old["pool"].shutdown(wait=False, cancel_futures=True)
    st.session_state.messages = []

def _render_hero() -> None:
    st.markdown("#### Ask anything about the Cryogen Free Measurement System")
    st.caption("Answers are retrieved from a PageIndex JSON tree of the CFMS internship report — IIT (BHU) Varanasi. No embeddings, no vector store, just vectorless RAG.")
    cards = [
    (":blue[:material/ac_unit:]",
     ":blue[**Cryogenics**]",
     "Cryo-cooler, cryostat & superconducting magnet."),

    (":orange[:material/thermostat:]",
     ":orange[**VTI & Control**]",
     "Temperature insert & electronics."),

    (":green[:material/fact_check:]",
     ":green[**Cited**]",
     "Every answer links to its source page."),
    ]
    for col, (icon, title, text) in zip(st.columns(3), cards):
        with col.container(border=True):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)

def _render_message(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["content"], icon=":material/error:")
        return

    st.write(msg["content"])

    if not msg.get("is_doc"):
        return

    sections = msg.get("sections", [])
    if sections:
        with st.expander(f"Sections used ({len(sections)})", icon=":material/menu_book:"):
            for s in sections:
                st.markdown(f":blue-badge[{s['id']}] **{s['title']}** :gray-badge[pages {s['pages']}]")

def _handle_prompt(app, prompt: str, welcome_slot) -> None:
    if not app.rate_limiter.acquire(blocking=False):
        st.warning("You're being rate-limited. Please wait a few seconds and try again.")
        return

    welcome_slot.empty()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant", avatar=AVATAR):
        with st.spinner("Reading document..."):
            try:
                result = app.ask(prompt, verbose=False)
                if result is None:
                    raise ValueError("The agent returned no structured response.")

                sections = []
                if result.is_doc_query:
                    for nid in result.node_ids:
                        node = app.nodes.get(nid)
                        if node:
                            sections.append({
                                "id":    nid,
                                "title": node["title"],
                                "pages": f"{node['start_index']}-{node['end_index']}",
                            })

                answer = result.answer.strip() or "I got an empty response. Please try rephrasing your question."
                msg = {
                    "role":     "assistant",
                    "content":  answer,
                    "is_doc":   result.is_doc_query,
                    "sections": sections,
                }
            except Exception as e:
                msg = {"role": "assistant", "content": f"{type(e).__name__}: {e}", "error": True}

        _render_message(msg)

    st.session_state.messages.append(msg)

def main() -> None:
    st.session_state.setdefault("messages", [])

    init   = _get_init()
    future = init["future"]

    try:
        error = future.exception() if future.done() else None
    except Exception as e:
        error = e

    app = future.result() if (future.done() and error is None) else None

    with st.sidebar:
        st.header(":material/settings: Status")
        if error is not None:
            st.error(f"Initialization failed: {error}", icon=":material/error:")
            st.button("Retry", icon=":material/refresh:", width="stretch", on_click=_on_retry)
        elif app is not None:
            with st.container(border=True):
                st.markdown(":green[:material/check_circle:] **Agent ready**")
                st.metric("Sections loaded", len(app.nodes), border=False)
            st.button(
                "New conversation",
                icon=":material/refresh:",
                width="stretch",
                on_click=_on_new_conversation,)
        else:
            with st.container(border=True):
                st.markdown(":orange[:material/hourglass_top:] **Agent starting…**")

    st.markdown("# :red[:material/smart_toy:] :red[CFMS RAG Agent]")

    welcome_slot = st.empty()
    has_chat_started = bool(st.session_state.messages) or ("_starter" in st.session_state)

    if not has_chat_started:
        with welcome_slot.container():
            _render_hero()
            if app is not None:
                STARTERS = [
                    "How does the cryo-cooler system work?",
                    "What is the Variable Temperature Insert (VTI) and how is it controlled?",
                    "What materials are used in the superconducting magnet?",
                    "Walk me through the procedure for operating the CFMS.",
                ]
                cols = st.columns(2)
                for i, starter in enumerate(STARTERS):
                    if cols[i % 2].button(starter, key=f"starter_{i}", use_container_width=True):
                        st.session_state["_starter"] = starter
                        st.rerun()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=AVATAR if msg["role"] == "assistant" else None):
                _render_message(msg)

    placeholder = (
        "Ask a question about the Cryogen Free Measurement System" if app is not None
        else "Initialization failed" if error is not None
        else "Agent is starting up…"
    )
    prompt = st.chat_input(placeholder, disabled=app is None)

    if not prompt and "_starter" in st.session_state:
        prompt = st.session_state.pop("_starter")

    if prompt and app is not None:
        _handle_prompt(app, prompt, welcome_slot)
    elif app is None and error is None:
        with st.sidebar, st.spinner("Starting in the background…"):
            try:
                future.result()
            except Exception:
                pass
        st.rerun()

main()