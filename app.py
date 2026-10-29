"""
TopicTrace: AI-Driven Automated Notes Generator
Streamlit Application
"""

import os
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

import importlib
import extractor
import llm_service
import exporter

# Hot-reload child modules in Streamlit
importlib.reload(extractor)
importlib.reload(llm_service)
importlib.reload(exporter)

from extractor import extract_text_from_pdf
from llm_service import (
    init_gemini_client,
    init_groq_client,
    parse_syllabus_to_topics,
    generate_notes_for_topic,
    generate_all_notes_parallel,
)
from exporter import export_markdown, export_pdf

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="TopicTrace | AI Study Notes Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .warning-banner {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        color: #92400e;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        font-weight: 600;
        margin: 1rem 0;
    }
    .metric-badge {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "syllabus_text" not in st.session_state:
    st.session_state.syllabus_text = None
if "syllabus_meta" not in st.session_state:
    st.session_state.syllabus_meta = None
if "textbook_text" not in st.session_state:
    st.session_state.textbook_text = None
if "textbook_meta" not in st.session_state:
    st.session_state.textbook_meta = None
if "parsed_units" not in st.session_state:
    st.session_state.parsed_units = None
if "generated_notes" not in st.session_state:
    st.session_state.generated_notes = ""
if "notes_editor_content" not in st.session_state:
    st.session_state.notes_editor_content = ""


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings & Keys")
    
    provider_choice = st.radio(
        "Choose AI Provider (100% Free)",
        options=["Google Gemini", "Groq"],
        help="Google Gemini offers large context windows. Groq offers lightning-fast (~500 tokens/sec) inference.",
    )

    if provider_choice == "Google Gemini":
        env_api_key = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Google Gemini API Key",
            value=env_api_key,
            type="password",
            help="Free key from Google AI Studio (aistudio.google.com).",
        )
        if not api_key_input:
            st.warning("⚠️ Gemini API Key required.")
            st.markdown("[👉 Get a free Gemini API Key](https://aistudio.google.com/app/apikey)")
        else:
            st.success("Gemini API Key detected!")

        model_choice = st.selectbox(
            "Select Gemini Model",
            options=["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-3.8-flash"],
            index=0,
            help="gemini-2.0-flash is recommended for ultra-fast, reliable study notes generation.",
        )
    else:  # Groq
        env_api_key = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=env_api_key,
            type="password",
            help="Free key from Groq Console (console.groq.com).",
        )
        if not api_key_input:
            st.warning("⚠️ Groq API Key required.")
            st.markdown("[👉 Get a free Groq API Key](https://console.groq.com/keys)")
        else:
            st.success("Groq API Key detected!")

        model_choice = st.selectbox(
            "Select Groq Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
            help="Llama 3.3 70B runs on Groq LPUs at ~500 tokens/sec with near-zero latency.",
        )

    def get_active_client(provider: str, key: str):
        if provider == "Groq":
            return init_groq_client(key)
        return init_gemini_client(key)

    st.markdown("---")
    st.subheader("📁 Document Upload")
    
    syllabus_file = st.file_uploader("1. Upload Syllabus (PDF)", type=["pdf"])
    textbook_file = st.file_uploader("2. Upload Textbook (PDF)", type=["pdf"])

    st.markdown("---")
    if st.button("🔄 Reset All Data", use_container_width=True):
        st.session_state.syllabus_text = None
        st.session_state.syllabus_meta = None
        st.session_state.textbook_text = None
        st.session_state.textbook_meta = None
        st.session_state.parsed_units = None
        st.session_state.generated_notes = ""
        st.session_state.notes_editor_content = ""
        st.rerun()


# Main Application Body
st.markdown('<div class="main-header">📚 TopicTrace: AI Study Notes Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract syllabus topics, cross-examine textbook content, and generate hallucination-free study notes.</div>', unsafe_allow_html=True)

# Extraction Trigger & Document Stats
if syllabus_file and textbook_file:
    # Check if files need processing
    need_extract_syllabus = (st.session_state.syllabus_text is None)
    need_extract_textbook = (st.session_state.textbook_text is None)

    if need_extract_syllabus or need_extract_textbook:
        with st.spinner("Extracting text from uploaded PDF files..."):
            if need_extract_syllabus:
                s_text, s_meta = extract_text_from_pdf(syllabus_file)
                st.session_state.syllabus_text = s_text
                st.session_state.syllabus_meta = s_meta

            if need_extract_textbook:
                t_text, t_meta = extract_text_from_pdf(textbook_file)
                st.session_state.textbook_text = t_text
                st.session_state.textbook_meta = t_meta

    # Display Document Overview
    col_s, col_t = st.columns(2)
    with col_s:
        s_meta = st.session_state.syllabus_meta
        st.info(f"**Syllabus:** {syllabus_file.name} | {s_meta['page_count']} pages | ~{s_meta['word_count']:,} words")
    with col_t:
        t_meta = st.session_state.textbook_meta
        st.info(f"**Textbook:** {textbook_file.name} | {t_meta['page_count']} pages | ~{t_meta['word_count']:,} words")

    # Step 1: Syllabus Topic Parsing
    st.markdown("### 1. Hierarchical Topic Parsing")
    
    if st.session_state.parsed_units is None:
        st.write("Extract the structured topic tree from your syllabus using Gemini.")
        if st.button("🚀 Parse Syllabus Topics", type="primary"):
            if not api_key_input:
                st.error("Please provide a Gemini API Key in the sidebar.")
            else:
                try:
                    status_banner = st.empty()
                    def parse_status_cb(msg: str):
                        status_banner.warning(msg)

                    with st.spinner(f"Analyzing syllabus hierarchy with {model_choice}..."):
                        client = get_active_client(provider_choice, api_key_input)
                        parsed = parse_syllabus_to_topics(
                            client,
                            st.session_state.syllabus_text,
                            model=model_choice,
                            provider=provider_choice,
                            status_callback=parse_status_cb,
                        )
                        status_banner.empty()
                        st.session_state.parsed_units = parsed
                        st.success("Syllabus topics parsed successfully!")
                        st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    if "503" in err_msg or "high demand" in err_msg.lower():
                        st.error(
                            f"Google Gemini servers are currently experiencing peak demand. "
                            f"Please select 'gemini-2.0-flash' in the sidebar dropdown and try again."
                        )
                    else:
                        st.error(f"Error parsing syllabus: {err_msg}")
    else:
        # Display Topic Tree with Selection
        st.write("Select the topics you want to generate notes for:")
        
        all_topics_to_generate = []
        for u_idx, unit_item in enumerate(st.session_state.parsed_units):
            unit_title = unit_item.get("unit", f"Unit {u_idx + 1}")
            with st.expander(f"📂 {unit_title}", expanded=True):
                topics = unit_item.get("topics", [])
                for t_idx, topic in enumerate(topics):
                    t_name = topic.get("name", "Unnamed Topic")
                    subtopics = topic.get("subtopics", [])
                    key = f"topic_sel_{u_idx}_{t_idx}"
                    
                    is_selected = st.checkbox(
                        f"**{t_name}**" + (f" ({', '.join(subtopics)})" if subtopics else ""),
                        value=True,
                        key=key,
                    )
                    if is_selected:
                        all_topics_to_generate.append({
                            "unit": unit_title,
                            "topic": t_name,
                            "subtopics": subtopics
                        })

        st.caption(f"Selected {len(all_topics_to_generate)} topic(s) for generation.")

        # Step 2: Note Generation
        st.markdown("### 2. Topic Mapping & Note Generation")
        
        col_gen_btn, col_turbo_toggle = st.columns([1, 1])
        with col_turbo_toggle:
            turbo_mode = st.checkbox(
                "⚡ Turbo Parallel Mode (4x Faster)",
                value=True,
                help="Generates notes concurrently across multiple threads with zero reasoning lag.",
            )
        
        with col_gen_btn:
            start_gen = st.button("📝 Generate Study Notes", type="primary", disabled=(len(all_topics_to_generate) == 0))

        if start_gen:
            if not api_key_input:
                st.error("Please provide a Gemini API Key in the sidebar.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    client = get_active_client(provider_choice, api_key_input)
                    total_topics = len(all_topics_to_generate)
                    doc_title_header = f"# Study Notes: {syllabus_file.name.replace('.pdf', '')}\n\n"

                    if turbo_mode:
                        status_text.text(f"⚡ Turbo Mode: Generating notes for {total_topics} topics in parallel...")

                        def on_parallel_progress(done: int, total: int, last_topic: str):
                            progress_bar.progress(done / total)
                            status_text.text(f"⚡ Turbo generating [{done}/{total}]: completed '{last_topic}'")

                        body_notes = generate_all_notes_parallel(
                            client=client,
                            topics_list=all_topics_to_generate,
                            textbook_text=st.session_state.textbook_text,
                            model=model_choice,
                            provider=provider_choice,
                            max_workers=4,
                            progress_callback=on_parallel_progress,
                        )
                        final_notes = doc_title_header + body_notes

                    else:
                        generated_sections = [doc_title_header]
                        current_unit = None
                        for idx, item in enumerate(all_topics_to_generate):
                            if item["unit"] != current_unit:
                                current_unit = item["unit"]
                                generated_sections.append(f"\n# {current_unit}\n")
                            
                            status_text.text(f"Cross-referencing textbook for [{idx + 1}/{total_topics}]: {item['topic']}...")
                            
                            def gen_status_cb(msg: str):
                                status_text.warning(f"[{idx + 1}/{total_topics}] {item['topic']}: {msg}")

                            note_content = generate_notes_for_topic(
                                client=client,
                                topic_name=item["topic"],
                                subtopics=item["subtopics"],
                                textbook_text=st.session_state.textbook_text,
                                model=model_choice,
                                provider=provider_choice,
                                status_callback=gen_status_cb,
                            )
                            generated_sections.append(note_content)
                            generated_sections.append("\n---\n")
                            progress_bar.progress((idx + 1) / total_topics)

                        final_notes = "\n".join(generated_sections)

                    status_text.empty()
                    progress_bar.empty()
                    
                    st.session_state.generated_notes = final_notes
                    st.session_state.notes_editor_content = final_notes
                    st.success("All notes generated successfully! Scroll down to review and edit.")
                except Exception as e:
                    st.error(f"Error during note generation: {str(e)}")

        # Step 3: Human Review & Live Editor
        if st.session_state.generated_notes:
            st.markdown("### 3. Human Review & Editing")
            
            # Mandatory AI Warning Banner
            st.markdown(
                '<div class="warning-banner">⚠️ AI-generated content. Please verify with your textbook.</div>',
                unsafe_allow_html=True,
            )

            col_edit, col_preview = st.columns([1, 1])

            with col_edit:
                st.subheader("✏️ Edit Notes (Markdown)")
                edited_text = st.text_area(
                    "You can modify, add, or delete any generated notes below:",
                    value=st.session_state.notes_editor_content,
                    height=600,
                    key="notes_editor_area",
                )
                st.session_state.notes_editor_content = edited_text

            with col_preview:
                st.subheader("👁️ Live Preview")
                preview_container = st.container(height=600)
                with preview_container:
                    st.markdown(st.session_state.notes_editor_content)

            # Step 4: Export Notes
            st.markdown("### 4. Export Notes")
            col_exp_md, col_exp_pdf = st.columns(2)

            with col_exp_md:
                md_bytes = export_markdown(st.session_state.notes_editor_content)
                st.download_button(
                    label="📥 Download as Markdown (.md)",
                    data=md_bytes,
                    file_name="study_notes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with col_exp_pdf:
                try:
                    pdf_bytes = export_pdf(
                        st.session_state.notes_editor_content,
                        title=f"Study Notes - {syllabus_file.name.replace('.pdf', '')}"
                    )
                    st.download_button(
                        label="📄 Download as PDF (.pdf)",
                        data=pdf_bytes,
                        file_name="study_notes.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error preparing PDF: {str(e)}")

else:
    # Empty State Guide
    st.info("👋 Welcome! Please upload both a **Syllabus PDF** and a **Textbook PDF** in the sidebar to begin.")
    
    with st.expander("📖 How It Works & Anti-Hallucination Pipeline", expanded=True):
        st.markdown(
            """
            1. **Upload Documents:** Upload your course syllabus and textbook PDFs.
            2. **Parse Hierarchy:** TopicTrace extracts the syllabus and identifies units, topics, and subtopics.
            3. **Strict Mapping:** For each topic, TopicTrace supplies the topic and textbook to Google Gemini with strict anti-hallucination instructions. If a topic is not in the textbook, the notes will explicitly flag:
               > ⚠️ **Missing from Textbook**: This topic is not covered in the provided textbook content.
            4. **Review & Edit:** Review the notes in a live side-by-side markdown editor.
            5. **Export:** Download your study notes as clean Markdown or a formatted PDF.
            """
        )
