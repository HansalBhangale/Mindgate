# ui_app.py
import streamlit as st
import pandas as pd
import io, re
from langchain_core.messages import HumanMessage, AIMessage
from router import get_router_chain, route_query
from bigquery_chain import BigQueryChain
from pdf_rag_chain import PDFRAGChain

# ---------------------------------------------------------------------
# 🧠 Initialize session state
# ---------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "router" not in st.session_state:
    st.session_state.router = get_router_chain()
if "bq_assistant" not in st.session_state:
    st.session_state.bq_assistant = BigQueryChain()
if "pdf_assistant" not in st.session_state:
    st.session_state.pdf_assistant = PDFRAGChain()

# ---------------------------------------------------------------------
# 🎨 Streamlit page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot")
st.caption("Ask about **customer transactions** (BigQuery) or **UPI process** (PDF).")

# ---------------------------------------------------------------------
# 🧩 Utility: Display results payload
# ---------------------------------------------------------------------
def display_results(payload: dict, message_index: int):
    """
    Intelligently displays results. If a DataFrame is present, it shows a
    paginated table inside an expander. The summary is written by the main loop.
    """
    dataframe = payload.get("dataframe")

    # Display the table only if the DataFrame exists, is not empty, and is not a single value (1x1).
    if dataframe is not None and not dataframe.empty and dataframe.shape != (1, 1):
        with st.expander(f"View Table ({len(dataframe)} rows)", expanded=True):
            rows_per_page = 10
            total_rows = len(dataframe)
            total_pages = (total_rows - 1) // rows_per_page + 1

            if total_rows <= rows_per_page:
                st.dataframe(dataframe, use_container_width=True)
            else:
                page = st.number_input("Page", 1, total_pages, 1, key=f"page_{message_index}")
                start_idx = (page - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                st.dataframe(dataframe.iloc[start_idx:end_idx], use_container_width=True)
                st.caption(f"Showing rows {start_idx + 1}–{min(end_idx, total_rows)} of {total_rows}.")

            csv = dataframe.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download as CSV", csv, "query_results.csv", "text/csv", key=f"dl_{message_index}")

# ---------------------------------------------------------------------
# 💬 Display the entire conversation on every rerun
# ---------------------------------------------------------------------
for i, (msg, payload) in enumerate(st.session_state.conversation):
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
        if payload:
            display_results(payload, i)

# ---------------------------------------------------------------------
# 🧠 Generate a new response if the last message is from the user
# ---------------------------------------------------------------------
if st.session_state.conversation and isinstance(st.session_state.conversation[-1][0], HumanMessage):
    user_query = st.session_state.conversation[-1][0].content
    messages_for_chain = [item[0] for item in st.session_state.conversation]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                destination = route_query(st.session_state.router, user_query)

                if destination == "BIGQUERY":
                    response_payload = st.session_state.bq_assistant.run(
                        user_query, messages_for_chain, return_full_history=False
                    )
                    summary = response_payload.get("summary", "Sorry, I couldn't process that.")
                    ai_message = AIMessage(content=summary)
                    st.session_state.conversation.append((ai_message, response_payload))

                else:  # PDF
                    answer = st.session_state.pdf_assistant.run(
                        user_query, messages_for_chain, return_full_history=False
                    )
                    ai_message = AIMessage(content=answer)
                    st.session_state.conversation.append((ai_message, None))

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.session_state.conversation.append((AIMessage(content=error_message), None))
            
            st.rerun()

# ---------------------------------------------------------------------
# 🚀 Handle user input
# ---------------------------------------------------------------------
if user_query := st.chat_input("Ask your question..."):
    # Append user message and immediately rerun to display it
    st.session_state.conversation.append((HumanMessage(content=user_query), None))
    st.rerun()
