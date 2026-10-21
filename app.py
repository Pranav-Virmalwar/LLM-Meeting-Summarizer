"""Streamlit application for summarizing meeting transcript files with OpenAI."""

import os
import tempfile
from pathlib import Path

import streamlit as st
import webvtt
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

def read_transcript(uploaded_file) -> str:
    """Extract text from a plain-text transcript or a WebVTT caption file."""
    if Path(uploaded_file.name).suffix.lower() == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vtt") as temporary_file:
        temporary_file.write(uploaded_file.getvalue())
        temporary_path = temporary_file.name
    try:
        captions = webvtt.read(temporary_path)
        return "\n".join(caption.text.strip() for caption in captions if caption.text.strip())
    finally:
        os.unlink(temporary_path)


def build_prompts(map_instruction: str, synthesis_instruction: str) -> tuple[PromptTemplate, PromptTemplate]:
    """Build map and reduce prompts, using defaults when fields are blank."""
    map_instruction = map_instruction.strip() or "Extract the main discussion topics, decisions made, and follow-up actions. Keep the summary concise and factual."
    synthesis_instruction = synthesis_instruction.strip() or "Combine the partial summaries into a coherent meeting summary. Include the overall themes, decisions, and action items."
    return (
        PromptTemplate.from_template("Transcript segment:\n{docs}\n\nTask:\n" + map_instruction),
        PromptTemplate.from_template("Partial summaries:\n{docs}\n\nTask:\n" + synthesis_instruction),
    )


st.set_page_config(page_title="Meeting Summary Generator", page_icon="📋", layout="wide", initial_sidebar_state="expanded")
st.title("Futurism's ConvoCraft: Crafting Clarity from Conversations")
st.divider()

with st.sidebar:
    st.title("Upload transcript")
    uploaded_file = st.file_uploader("Meeting transcript", type=["vtt", "txt"])
    map_instruction = st.text_area("Extraction prompt (optional)", help="Applied to each transcript segment.")
    synthesis_instruction = st.text_area("Synthesis prompt (optional)", help="Applied to combine the segment summaries.")
    generate = st.button("Generate summary", type="primary")

if generate:
    if uploaded_file is None:
        st.sidebar.error("Please upload a .vtt or .txt transcript.")
        st.stop()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not configured. Add it to your environment and restart the app.")
        st.stop()
    try:
        transcript = read_transcript(uploaded_file)
        if not transcript:
            st.error("The uploaded file did not contain readable transcript text.")
            st.stop()
        splitter = CharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-3.5-turbo", chunk_size=3000, chunk_overlap=200)
        documents = splitter.split_documents([Document(page_content=transcript)])
        map_prompt, combine_prompt = build_prompts(map_instruction, synthesis_instruction)
        llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo", openai_api_key=api_key)
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
        with st.spinner("Summarizing the meeting transcript..."):
            summary = chain.run(documents)
        st.subheader("Meeting summary")
        st.markdown(summary)
    except Exception as error:
        st.error(f"Unable to summarize the transcript: {error}")
else:
    st.info("Upload a transcript and select **Generate summary** to begin.")
