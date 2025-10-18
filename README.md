# Hybrid Data Chatbot

This project is a conversational AI chatbot that can answer questions from two distinct data sources: a **Google BigQuery** database containing transactional data and a local **PDF document** processed via a Retrieval-Augmented Generation (RAG) pipeline.

The chatbot uses a router to dynamically determine the user's intent and query the appropriate data source, making it a powerful tool for hybrid data exploration. It features both a command-line interface and a rich web UI built with Streamlit.

## Features

- **Dual Data Sources**: Seamlessly queries both a structured BigQuery database and unstructured PDF documents.
- **LLM-Powered Routing**: Intelligently routes user questions to the correct data source based on their content.
- **Natural Language to SQL**: Translates natural language questions into complex BigQuery SQL queries.
- **Retrieval-Augmented Generation (RAG)**: Uses a FAISS vector store and LangGraph to provide context-aware answers from a PDF document.
- **Conversational Memory**: Maintains conversation history to answer follow-up questions accurately.
- **Dual Interfaces**: Includes a simple CLI (`app.py`) and an interactive Streamlit web app (`app_ui.py`) with paginated tables and CSV downloads.

## Architecture

The application follows a simple yet powerful "Router-Agent" pattern:

1.  **User Input**: The user asks a question in natural language.
2.  **Router**: An LLM chain analyzes the question to determine its topic (e.g., "customer transactions" vs. "UPI process").
3.  **Dispatch**: The router directs the question to one of two specialized chains:
    - **BigQuery Chain**: For questions about structured data. It generates a SQL query, executes it on BigQuery, and summarizes the result.
    - **PDF RAG Chain**: For questions about the PDF content. It retrieves relevant text chunks from a FAISS vector store and generates an answer.
4.  **Response**: The final, summarized answer is presented to the user.

```
User Query -> [Router] -> | -> [BigQuery Chain] -> Google BigQuery
                         |
                         | -> [PDF RAG Chain] -> FAISS Vector Store
```

## Prerequisites

- Python 3.9+
- A Google Cloud Platform (GCP) project with BigQuery enabled.
- An OpenAI API key.

## Setup and Installation

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd <repository-name>
```

**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**

Create a file named `.env` in the root of the project and add the following, replacing the placeholder values:

```env
# OpenAI API Key
OPENAI_API_KEY="sk-..."

# Google Cloud Configuration
GCP_PROJECT_ID="your-gcp-project-id"
BIGQUERY_DATASET="your_bigquery_dataset_name"

# Path to your GCP service account JSON file
# Ensure this file has permissions for BigQuery
GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"

# Path to the PDF file you want to index
SOURCE_PDF_PATH="data/your_document.pdf"
```

**5. Set up Google Cloud Authentication**

Ensure you have authenticated your local environment with GCP. The simplest way is to use the gcloud CLI:
```bash
gcloud auth application-default login
```
This will make your credentials available to the application. Alternatively, ensure the `GOOGLE_APPLICATION_CREDENTIALS` path in your `.env` file is correct.

**6. Index Your PDF Document**

Place the PDF you want the chatbot to read into the `data/` directory. Then, run the indexer script to create the vector store. This is a one-time setup step.

```bash
python pdf_indexer.py
```
This will create a `vector_store_openai` directory containing the FAISS index.

## Running the Application

You can interact with the chatbot using either the command-line interface or the Streamlit web UI.

**CLI Version**
```bash
python app.py
```

**Streamlit Web UI**
```bash
streamlit run app_ui.py
```

## Project Structure

```
.
├── app.py                  # Main entrypoint for the CLI application.
├── app_ui.py               # Main entrypoint for the Streamlit web UI.
├── bigquery_chain.py       # Logic for the BigQuery agent (NL-to-SQL, summarization).
├── pdf_rag_chain.py        # Logic for the PDF RAG agent (retrieval, generation).
├── router.py               # LLM-based router to select the correct chain.
├── pdf_indexer.py          # Script to create the vector store from a PDF.
├── config.py               # Handles loading of environment variables.
├── requirements.txt        # Python package dependencies.
├── .env                    # Local file for storing secrets (not committed).
├── data/                   # Directory to store source PDF files.
└── vector_store_openai/    # Directory containing the generated FAISS index.
```
