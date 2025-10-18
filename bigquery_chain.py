from google.cloud import bigquery
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import GCP_PROJECT_ID, BIGQUERY_DATASET, OPENAI_API_KEY

# ... (rest of the file remains the same)
def get_bigquery_chain():
    """Initializes the full chain for handling BigQuery requests."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)

    # This schema is based on the reference CSV files you provided.
    schema = f"""
    You have access to two tables in the `{BIGQUERY_DATASET}` dataset:
    1. `customers` table with columns: customer_id (INTEGER), customer_name (STRING), email (STRING), phone_number (STRING), address (STRING), customer_since (DATE).
    2. `transactions` table with columns: transaction_id (INTEGER), customer_id (INTEGER), account_number (STRING), account_type (STRING), transaction_timestamp (TIMESTAMP), transaction_amount (FLOAT), transaction_type (STRING, values are 'Credit' or 'Debit'), counterparty_name (STRING), counterparty_account (STRING).
    """

    def _execute_query(sql_query: str) -> str:
        """Helper function to run the SQL query and format the result."""
        try:
            clean_sql = sql_query.strip().replace("```sql", "").replace("```", "")
            print(f"--- Executing SQL: ---\n{clean_sql}\n--------------------")
            
            query_job = bq_client.query(clean_sql)
            results = query_job.to_dataframe()
            
            if results.empty:
                return "The query executed successfully but returned no results."
            return results.to_string()
        except Exception as e:
            return f"An error occurred while executing the BigQuery query: {e}"

    def run_full_chain(question: str) -> str:
        # Step 1: Generate SQL from the user's question
        sql_prompt = PromptTemplate.from_template(
            """You are a BigQuery SQL expert. Given a user question and the database schema, write a valid BigQuery SQL query to answer it.
            Only output the SQL query and nothing else. Ensure the query references tables with the full dataset name, like `{dataset}.customers`.

            Schema: {schema}
            Question: {question}
            SQL Query:"""
        )
        sql_chain = sql_prompt | llm
        generated_sql = sql_chain.invoke({"schema": schema, "question": question, "dataset": BIGQUERY_DATASET}).content

        # Step 2: Execute the generated SQL
        query_result = _execute_query(generated_sql)

        # Step 3: Synthesize a final, human-readable answer
        synthesis_prompt = PromptTemplate.from_template(
            """You are a helpful assistant. Based on the user's original question and the data retrieved from the database, provide a clear and friendly answer.

            Original Question: {question}
            Database Result:
            {result}
            
            Answer:"""
        )
        synthesis_chain = synthesis_prompt | llm
        return synthesis_chain.invoke({"question": question, "result": query_result}).content

    return run_full_chain