# Objective
Build a fully functional, simple MVP of an "AI-Driven Automated Notes Generator" web app using Python and Streamlit. 

The app should take a Syllabus PDF and a Textbook PDF, extract the topics, map them to the textbook content, generate study notes using an LLM, allow the user to edit the notes, and export them. 

# Tech Stack
- Choose the one where we can develop a mvp asap
- **Export:** `FPDF` or `markdown-pdf` to export the final notes.

# Core Features & Workflow (Must be fully functional)

1. **Upload (Sidebar or Main Page):**
   - File uploader for Syllabus (PDF).
   - File uploader for Textbook (PDF).

2. **Processing & Extraction:**
   - Extract text from the Syllabus.
   - Prompt the LLM to parse the raw syllabus text into a clean, hierarchical list of topics and subtopics (JSON format).
   - Extract text from the Textbook.

3. **Topic Mapping & Note Generation:**
   - For each syllabus topic, pass the topic name and the full textbook text (or a relevant chunk) to the LLM.
   - **Crucial Prompting:** Instruct the LLM to generate study notes *strictly* based on the provided textbook text. It must explicitly state if a topic is missing from the textbook to prevent hallucination.
   - Organize the notes clearly under their respective headings.

4. **Human Review (UI):**
   - Display the generated notes on the screen.
   - Use a rich text editor so the user can manually edit, add, or delete the generated notes.
   - Add a warning banner: "AI-generated content. Please verify with your textbook."

5. **Export:**
   - Add a button to download the final, edited notes as a clean PDF or Markdown file.

# Execution Instructions
1. List the required `pip install` packages (e.g., streamlit, pdfplumber, openai, fpdf).
2. Generate the complete, runnable code (e.g., `app.py`).
3. Keep the code clean, modular (separate functions for parsing, generating, exporting), and well-commented so I can easily run it locally. 

Focus entirely on making the end-to-end pipeline work smoothly.
