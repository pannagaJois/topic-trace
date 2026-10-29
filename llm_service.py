"""
LLM Service Module for TopicTrace.
Supports Local Offline Model (Qwen2.5-1.5B via llama-cpp-python), Google Gemini, and Groq.
Local model requires zero API keys, zero network requests, and runs 100% privately on your CPU.
"""

import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import httpx

# Optional Cloud Providers
from google import genai
from google.genai import types as gemini_types
from groq import Groq

# Local GGUF Engine
from llama_cpp import Llama

# Global singleton for cached local LLM instance
_LOCAL_LLM_INSTANCE = None
DEFAULT_LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")


def get_local_llm(model_path: str = DEFAULT_LOCAL_MODEL_PATH, n_ctx: int = 4096, n_threads: int = 6) -> Llama:
    """
    Returns a cached instance of the local Llama GGUF model.
    """
    global _LOCAL_LLM_INSTANCE
    if _LOCAL_LLM_INSTANCE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model not found at {model_path}. Please download it first.")
        _LOCAL_LLM_INSTANCE = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
    return _LOCAL_LLM_INSTANCE


def init_gemini_client(api_key: Optional[str] = None, timeout: float = 90.0) -> genai.Client:
    """
    Initializes and returns a Google GenAI Client with an explicit network timeout.
    """
    if not api_key:
        raise ValueError("Gemini API Key is required. Please provide it in the sidebar or via GEMINI_API_KEY environment variable.")
    return genai.Client(
        api_key=api_key,
        http_options=gemini_types.HttpOptions(timeout=timeout),
    )


def init_groq_client(api_key: Optional[str] = None, timeout: float = 60.0) -> Groq:
    """
    Initializes and returns a Groq Client (100% free tier at console.groq.com).
    """
    if not api_key:
        raise ValueError("Groq API Key is required. Please provide it in the sidebar or via GROQ_API_KEY environment variable.")
    return Groq(api_key=api_key, timeout=timeout)


def _clean_json_response(raw_text: str) -> str:
    """
    Extracts JSON substring from LLM response, stripping markdown backticks if present.
    """
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    
    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        return text[start_bracket : end_bracket + 1]

    return text


def get_relevant_textbook_context(
    textbook_text: str,
    topic_name: str,
    subtopics: List[str],
    max_chars: int = 80000
) -> str:
    """
    Extracts matching textbook pages and neighbor context to fit comfortably in local context.
    """
    if len(textbook_text) <= max_chars:
        return textbook_text

    if "--- Page " in textbook_text:
        pages = textbook_text.split("--- Page ")
        keywords = [k.lower() for k in [topic_name] + subtopics if len(k) > 2]
        
        matched_indices = set()
        for idx, page in enumerate(pages):
            p_lower = page.lower()
            if any(kw in p_lower for kw in keywords):
                for offset in (-1, 0, 1):
                    neighbor = idx + offset
                    if 0 <= neighbor < len(pages):
                        matched_indices.add(neighbor)

        if matched_indices:
            selected_pages = [pages[i] for i in sorted(matched_indices)]
            combined = "--- Page " + "--- Page ".join(selected_pages)
            return combined[:max_chars]
        else:
            return ("--- Page " + "--- Page ".join(pages[:5]))[:max_chars]

    return textbook_text[:max_chars]


def resilient_generate_content(
    client: Any,
    contents: str,
    primary_model: str = "local-qwen-1.5b",
    provider: str = "Local Model",
    config: Optional[Any] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    max_tokens: int = 768,
) -> str:
    """
    Executes text generation using Local Model, Google Gemini, or Groq.
    """
    # 1. Local Offline Model
    if provider.startswith("Local"):
        local_llm = client if isinstance(client, Llama) else get_local_llm()
        chatml_prompt = f"<|im_start|>user\n{contents}<|im_end|>\n<|im_start|>assistant\n"
        out = local_llm(
            chatml_prompt,
            max_tokens=max_tokens,
            temperature=0.15,
            stop=["<|im_end|>", "<|endoftext|>", "</s>"]
        )
        return out["choices"][0]["text"].strip()

    # 2. Groq Cloud
    if provider == "Groq":
        groq_candidates = [primary_model]
        for alt in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            if alt not in groq_candidates:
                groq_candidates.append(alt)

        last_error = None
        for model_name in groq_candidates:
            for attempt in range(2):
                try:
                    chat_resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": contents}],
                        temperature=0.2,
                    )
                    return chat_resp.choices[0].message.content or ""
                except Exception as e:
                    last_error = e
                    if status_callback:
                        status_callback(f"⚠️ Groq {model_name} issue: {str(e)[:60]}... Retrying...")
                    time.sleep(2)
        raise last_error

    # 3. Google Gemini Cloud
    gemini_candidates = [primary_model]
    for backup in ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-3.8-flash"]:
        if backup not in gemini_candidates:
            gemini_candidates.append(backup)

    last_error = None
    current_config = config

    for model_name in gemini_candidates:
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=current_config,
                )
                return response.text or ""

            except Exception as e:
                err_str = str(e).lower()
                is_timeout = isinstance(e, (httpx.TimeoutException, TimeoutError)) or "timeout" in err_str
                is_transient = (
                    is_timeout
                    or "503" in err_str
                    or "unavailable" in err_str
                    or "high demand" in err_str
                    or "429" in err_str
                    or "resource_exhausted" in err_str
                )
                last_error = e

                if "thinking" in err_str and current_config and getattr(current_config, "thinking_config", None):
                    current_config = gemini_types.GenerateContentConfig(temperature=0.2)
                    continue

                if is_transient:
                    wait_time = 2 * (attempt + 1)
                    if attempt < max_attempts - 1:
                        if status_callback:
                            reason = "timed out" if is_timeout else "busy"
                            status_callback(f"⚠️ {model_name} was {reason}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        if status_callback and model_name != gemini_candidates[-1]:
                            next_model = gemini_candidates[gemini_candidates.index(model_name) + 1]
                            status_callback(f"⚠️ {model_name} not responding. Switching to fast backup: {next_model}...")
                        time.sleep(1)
                else:
                    raise e

    raise last_error if last_error else RuntimeError("Model generation failed across all available fallback models.")


def parse_syllabus_to_topics(
    client: Any,
    syllabus_text: str,
    model: str = "local-qwen-1.5b",
    provider: str = "Local Model",
    status_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Prompts the LLM to parse raw syllabus text into a clean hierarchical JSON list of topics/subtopics.
    """
    max_chars = 25000 if provider.startswith("Local") else 40000
    cleaned_syllabus = syllabus_text[:max_chars].strip()

    prompt = f"""You are an expert curriculum analyzer. Parse the following syllabus into a JSON array of units and topics.
Output ONLY a valid JSON array matching this exact schema:
[
  {{
    "unit": "Unit/Module title or Chapter name",
    "topics": [
      {{
        "name": "Topic title",
        "subtopics": ["Subtopic 1", "Subtopic 2"]
      }}
    ]
  }}
]

SYLLABUS TEXT:
{cleaned_syllabus}
"""

    config = gemini_types.GenerateContentConfig(
        temperature=0.1,
        thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
    ) if provider == "Google Gemini" else None

    response_text = resilient_generate_content(
        client=client,
        contents=prompt,
        primary_model=model,
        provider=provider,
        config=config,
        status_callback=status_callback,
        max_tokens=768,
    )

    clean_json = _clean_json_response(response_text or "[]")
    try:
        parsed = json.loads(clean_json)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "units" in parsed:
            return parsed["units"]
        elif isinstance(parsed, dict) and "topics" in parsed:
            return [{"unit": "General Topics", "topics": parsed["topics"]}]
        else:
            return [{"unit": "Extracted Topics", "topics": [{"name": str(k), "subtopics": v if isinstance(v, list) else []} for k, v in parsed.items()]}]
    except json.JSONDecodeError:
        return [
            {
                "unit": "Overview",
                "topics": [
                    {
                        "name": "Parsed Content",
                        "subtopics": [line.strip("- *• ") for line in response_text.splitlines() if line.strip()][:15]
                    }
                ]
            }
        ]


def generate_notes_for_topic(
    client: Any,
    topic_name: str,
    subtopics: List[str],
    textbook_text: str,
    model: str = "local-qwen-1.5b",
    provider: str = "Local Model",
    status_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
) -> str:
    """
    Generates study notes for a single topic and its subtopics, strictly grounded on the textbook text.
    """
    subtopics_str = ", ".join(subtopics) if subtopics else "None specified"
    max_context_chars = 30000 if provider.startswith("Local") else 100000
    relevant_context = get_relevant_textbook_context(textbook_text, topic_name, subtopics, max_chars=max_context_chars)

    prompt = f"""You are an academic study notes generator.
Generate study notes for the topic '{topic_name}' strictly based on the provided Textbook Text.

STRICT GROUNDING RULES:
1. Use ONLY facts directly present in the textbook text below.
2. If the topic is missing from the textbook text, output:
> ⚠️ **Missing from Textbook**: This topic is not covered in the provided textbook content.

TOPIC: {topic_name}
SUBTOPICS: {subtopics_str}

TEXTBOOK TEXT:
{relevant_context}

FORMAT YOUR RESPONSE AS:
## {topic_name}
- **Overview & Definition**: Key definition from the text.
- **Core Concepts**: Detailed bullet points from the text.
- **Key Takeaways**: Concise summary.
"""

    config = gemini_types.GenerateContentConfig(
        temperature=0.2,
        thinking_config=gemini_types.ThinkingConfig(thinking_budget=0),
    ) if provider == "Google Gemini" else None

    response_text = resilient_generate_content(
        client=client,
        contents=prompt,
        primary_model=model,
        provider=provider,
        config=config,
        status_callback=status_callback,
        max_tokens=600,
    )

    return (response_text or "").strip()


def generate_all_notes_parallel(
    client: Any,
    topics_list: List[Dict[str, Any]],
    textbook_text: str,
    model: str = "local-qwen-1.5b",
    provider: str = "Local Model",
    max_workers: int = 2,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Generates notes for all selected topics.
    For local CPU models, uses 2 workers with multi-threading to prevent CPU thrashing.
    """
    total = len(topics_list)
    results = [None] * total

    def _worker(item_with_idx):
        idx, item = item_with_idx
        note = generate_notes_for_topic(
            client=client,
            topic_name=item["topic"],
            subtopics=item["subtopics"],
            textbook_text=textbook_text,
            model=model,
            provider=provider,
        )
        return idx, item["topic"], note

    # Local model is optimized with 1-2 workers on CPU
    actual_workers = 1 if provider.startswith("Local") else max_workers

    completed_count = 0
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = [executor.submit(_worker, (i, it)) for i, it in enumerate(topics_list)]
        for f in as_completed(futures):
            idx, topic_name, note = f.result()
            results[idx] = note
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total, topic_name)

    assembled_sections = []
    current_unit = None
    for idx, item in enumerate(topics_list):
        if item["unit"] != current_unit:
            current_unit = item["unit"]
            assembled_sections.append(f"\n# {current_unit}\n")
        assembled_sections.append(results[idx] or f"## {item['topic']}\n*(Could not generate notes)*")
        assembled_sections.append("\n---\n")

    return "\n".join(assembled_sections)
