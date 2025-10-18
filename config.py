import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Google Cloud Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = "us-central1"
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET")

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Service Account Configuration ---
# This line is crucial. It loads the path from .env into the os environment
# so that Google's libraries can find and use it automatically.
SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if SERVICE_ACCOUNT_PATH:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH

# --- File Path Configuration ---
PDF_PATH = os.path.join("data", "UPI Transaction Process Explained.pdf")
VECTOR_STORE_PATH_OPENAI = "vector_store_openai"

# --- Validation ---
if not all([GCP_PROJECT_ID, BIGQUERY_DATASET, OPENAI_API_KEY]):
    raise ValueError(
        "One or more required environment variables are missing. "
        "Please check your .env file for: \n"
        "GCP_PROJECT_ID\n"
        "BIGQUERY_DATASET\n"
        "OPENAI_API_KEY\n"
    )

if not SERVICE_ACCOUNT_PATH:
    print("Warning: GOOGLE_APPLICATION_CREDENTIALS is not set. BigQuery will use default credentials.")