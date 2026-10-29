"""
LLM Service Module for TopicTrace.
Handles syllabus parsing into hierarchical JSON and textbook-grounded study note generation using Google Gemini.
Includes automatic retries, backoff, and fallback models for transient high-demand (503/429) errors.
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Callable
from google import genai
from google.genai import types, errors


def init_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes and returns a Google GenAI Client.
    """
    if not api_key:
        raise ValueError("Gemini API Key is required. Please provide it in the sidebar or via GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


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


def resilient_generate_content(
    client: genai.Client,
    contents: str,
    primary_model: str = "gemini-3.8-flash",
    config: Optional[types.GenerateContentConfig] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes generate_content with automatic retries, exponential backoff,
    and automatic fallback to alternative Gemini models if Google servers report 503 UNAVAILABLE or 429.
    """
    # Priority order of models to try if the primary encounters high demand
    candidates = [primary_model]
    for backup in ["gemini-2.0-flash", "gemini-3.8-pro", "gemini-1.5-flash"]:
        if backup not in candidates:
            candidates.append(backup)

    last_error = None

    for model_name in candidates:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e).lower()
                is_transient = (
                    "503" in err_str
                    or "unavailable" in err_str
                    or "high demand" in err_str
                    or "429" in err_str
                    or "resource_exhausted" in err_str
                )
                last_error = e

                if is_transient:
                    wait_time = (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        if status_callback:
                            status_callback(f"⚠️ {model_name} is experiencing high demand. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # If retries for this model are exhausted, announce switch to backup
                        if status_callback and model_name != candidates[-1]:
                            next_model = candidates[candidates.index(model_name) + 1]
                            status_callback(f"⚠️ {model_name} busy. Automatically switching to backup: {next_model}...")
                        time.sleep(1)
                else:
                    # Non-transient errors (such as 401 invalid key, bad request) raise immediately
                    raise e

    raise last_error if last_error else RuntimeError("Model generation failed across all available fallback models.")


def parse_syllabus_to_topics(
    client: genai.Client,
    syllabus_text: str,
    model: str = "gemini-2.0-flash",
    status_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Prompts the LLM to parse raw syllabus text into a clean hierarchical list of topics/subtopics.
    """
    prompt = f"""
You are an expert curriculum analyzer.
Your task is to parse the following raw syllabus text into a clean, hierarchical JSON list of units/modules, topics, and subtopics.

Strict Requirements:
1. Output ONLY a valid JSON array. Do not include introductory or concluding conversational text.
2. The JSON schema must strictly be:
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
3. If no units or modules are explicitly named, group topics under "General Topics".
4. Ensure each topic has a clear "name" and an array of "subtopics" (can be empty if none specified).

RAW SYLLABUS TEXT:
{syllabus_text}
"""

    response_text = resilient_generate_content(
        client=client,
        contents=prompt,
        primary_model=model,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
        status_callback=status_callback,
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
    client: genai.Client,
    topic_name: str,
    subtopics: List[str],
    textbook_text: str,
    model: str = "gemini-2.0-flash",
    status_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
) -> str:
    """
    Generates study notes for a single topic and its subtopics, strictly grounded on the textbook text.
    Explicitly flags if the topic is missing from the textbook.
    """
    subtopics_str = ", ".join(subtopics) if subtopics else "None specified"

    prompt = f"""
You are an expert academic tutor and study notes compiler.
Generate comprehensive, high-quality study notes for the specified topic and subtopics SOLELY and STRICTLY based on the provided Textbook Text.

TOPIC: {topic_name}
SUBTOPICS: {subtopics_str}

CRUCIAL GROUNDING & ANTI-HALLUCINATION RULES:
1. STRICT TEXTBOOK ADHERENCE: Use ONLY facts, definitions, explanations, formulas, mechanisms, and examples directly present in the Textbook Text below.
2. NO OUTSIDE KNOWLEDGE: Absolutely do NOT bring in outside knowledge or assumptions not found in the textbook text.
3. EXPLICIT MISSING WARNING:
   If the topic or any subtopic is NOT covered or absent in the textbook text, you MUST clearly state:
   > ⚠️ **Missing from Textbook**: This topic is not covered in the provided textbook content.
   Do NOT fabricate an explanation to fill the gap.
4. PARTIAL COVERAGE: If only parts of the topic/subtopics are covered, provide notes for the covered parts and explicitly flag which subtopics are missing.

NOTES STRUCTURE (Use clear Markdown):
## {topic_name}
- **Overview & Core Definition**: Concise definition grounded directly in the text.
- **Detailed Concepts & Notes**: Clear bullet points with in-depth explanation from the textbook.
- **Key Principles / Formulas / Steps**: (If covered in the text).
- **Textbook Examples**: (Only if mentioned in the text).
- **Key Takeaways**: High-yield summary points.

TEXTBOOK TEXT:
{textbook_text}
"""

    response_text = resilient_generate_content(
        client=client,
        contents=prompt,
        primary_model=model,
        config=types.GenerateContentConfig(
            temperature=0.3,
        ),
        status_callback=status_callback,
    )

    return (response_text or "").strip()
