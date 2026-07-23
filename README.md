# Label Change Impact Agent

An **Agentic AI system for pharmaceutical regulatory operations** that analyzes changes between an existing product label and an updated product label, identifies potentially affected downstream content, assesses the actual impact, and generates an actionable **Change-Impact Task List**.

The system combines **Llama 3.1**, **Ollama**, **LangGraph**, **RAG**, **FAISS**, **Sentence Transformers**, and **Streamlit** into an end-to-end AI workflow.

---

## Problem Statement

Pharmaceutical product label changes can affect multiple downstream resources, including:

- Artwork and packaging
- Website content
- Training materials
- Country regulatory submissions

Manually reviewing every downstream resource can be time-consuming and may lead to missed dependencies.

The **Label Change Impact Agent** automates this workflow by analyzing label changes and identifying which downstream resources require review or updates.

---

## Key Features

- Upload old and updated pharmaceutical label PDFs
- Extract text automatically from both labels
- Detect meaningful safety and regulatory changes
- Calculate overall label similarity
- Search downstream resources using semantic retrieval
- Perform RAG-based impact analysis
- Assess actual downstream impact using a local LLM
- Assign impact priorities
- Generate actionable tasks
- Suggest responsible teams for each task
- Track the complete agent workflow
- Download the final impact analysis report

---

## Agentic AI Workflow

The application uses a LangGraph-based workflow:

```text
START
  |
  v
Read Product Labels
  |
  v
Detect Safety and Regulatory Changes
  |
  v
Retrieve Potentially Affected Downstream Resources
  |
  v
Assess Actual Downstream Impact
  |
  v
Generate Change-Impact Task List
  |
  v
END
```

Each stage operates as a node in the agent workflow and shares information through a common agent state.

---

## System Architecture

### 1. Label Reader

Reads the old and updated pharmaceutical label PDFs and extracts their textual content.

### 2. Change Detector

Compares the two labels and uses **Llama 3.1** to identify meaningful safety and regulatory changes.

The output includes information such as:

- Change type
- Affected section
- Old information
- New information
- Safety significance
- Potential downstream impact

### 3. RAG Retrieval

The detected changes are converted into semantic embeddings using a Sentence Transformer model.

A **FAISS vector store** searches the downstream content inventory and retrieves potentially affected resources.

The sample downstream inventory contains:

- Artwork
- Website content
- Training material
- Country regulatory submission

### 4. Impact Assessment

Llama 3.1 evaluates the retrieved resources against the detected label changes and determines:

- Whether the resource is actually impacted
- Impact priority
- Reason for the impact
- Recommended action

### 5. Task Generator

The final agent converts actionable impacts into a structured **Change-Impact Task List** containing:

- Task ID
- Affected resource
- Category
- Priority
- Action required
- Reason
- Suggested owner
- Status

---

## Technology Stack

- **Python**
- **LangGraph** — Agent workflow orchestration
- **LangChain Ollama** — Local LLM integration
- **Ollama**
- **Llama 3.1**
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Semantic embeddings
- **Hugging Face**
- **Streamlit** — User interface
- **PyMuPDF** — PDF text extraction

---

## Project Structure

```text
Label-Change-Impact-Agent-AgenticAI/
|
|-- agent/
|   |-- state.py
|   |-- nodes.py
|   `-- graph.py
|
|-- data/
|   |-- artwork/
|   |   `-- metformin_packaging.txt
|   |
|   |-- country_submissions/
|   |   `-- metformin_regulatory_submission.txt
|   |
|   |-- training/
|   |   `-- metformin_safety_training.txt
|   |
|   `-- website/
|       `-- metformin_product_page.txt
|
|-- rag/
|   |-- embeddings.py
|   |-- inventory_loader.py
|   `-- vector_store.py
|
|-- tools/
|   |-- label_reader.py
|   |-- change_detector.py
|   |-- impact_assessment.py
|   `-- task_generator.py
|
|-- reports/
|
|-- app.py
|-- requirements.txt
|-- .gitignore
|
|-- test_agent.py
|-- test_change_detector.py
|-- test_impact_assessment.py
|-- test_label_reader.py
|-- test_rag.py
`-- test_task_generator.py
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Label-Change-Impact-Agent-AgenticAI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama on your system and make sure it is running.

### 5. Download Llama 3.1

```bash
ollama pull llama3.1
```

Verify the model:

```bash
ollama run llama3.1
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

```text
http://localhost:8501
```

Then:

1. Upload the old product label PDF.
2. Upload the updated product label PDF.
3. Click **Run Impact Analysis**.
4. Wait for the Agentic AI workflow to complete.
5. Review the detected changes and retrieved downstream resources.
6. Review the downstream impact assessment.
7. Review the generated Change-Impact Task List.
8. Download the final impact report.

---

## Example Agent Workflow

For a detected label update involving a new renal-function monitoring requirement, the system can:

1. Detect the change in the updated product label.
2. Search the downstream content inventory.
3. Retrieve semantically related regulatory and safety resources.
4. Determine which resources require action.
5. Assign priorities to the impacted resources.
6. Generate tasks for the appropriate teams.

An example generated task may contain:

```text
TASK ID: TASK-001

AFFECTED RESOURCE:
metformin_safety_training

CATEGORY:
Training Material

PRIORITY:
High

ACTION REQUIRED:
Review and update the training material to reflect the new monitoring requirement.

SUGGESTED OWNER:
Training Team

STATUS:
Pending
```

---

## Privacy and Local AI

The application uses **Llama 3.1 through Ollama**, allowing LLM inference to run locally.

This architecture is useful for demonstrating how Agentic AI workflows can be designed for document-intensive regulatory use cases while reducing reliance on external LLM APIs.

---

## Current Scope

This project is a prototype demonstrating an Agentic AI workflow for pharmaceutical label-change impact analysis.

The included downstream documents form a sample content inventory for demonstrating semantic retrieval and impact assessment. In a production environment, the same architecture could be integrated with larger enterprise document repositories, regulatory content management systems, and workflow platforms.

AI-generated impact assessments should be reviewed by appropriate regulatory and subject-matter experts before operational use.

---

## Future Enhancements

Potential improvements include:

- Larger enterprise downstream content inventories
- Persistent vector databases
- Structured JSON outputs from agent nodes
- Human-in-the-loop approval workflows
- Automatic task assignment and tracking
- Regulatory submission management integrations
- Version history and audit trails
- Authentication and role-based access
- OCR support for scanned product labels
- More advanced document-level change visualization

---

## Disclaimer

This project is intended as an **AI prototype and demonstration system**. It is not intended to replace qualified regulatory, medical, legal, or quality review.

---

## Author

**Vibha Sree Gowda**