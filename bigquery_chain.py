from typing import Dict, Any, TypedDict, Annotated, Sequence, Tuple
from google.cloud import bigquery
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from config import GCP_PROJECT_ID, BIGQUERY_DATASET, OPENAI_API_KEY
import pandas as pd

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sql_result: str
    generated_sql: str


class BigQueryChain:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        self.bq_client = bigquery.Client(project=GCP_PROJECT_ID)
        self.schema = f"""
        You have access to two tables in the `{BIGQUERY_DATASET}` dataset:
        1. `customers` table with columns: customer_id (INTEGER), customer_name (STRING), email (STRING), phone_number (STRING), address (STRING), customer_since (DATE).
        2. `transactions` table with columns: transaction_id (INTEGER), customer_id (INTEGER), account_number (STRING), account_type (STRING), transaction_timestamp (TIMESTAMP), transaction_amount (FLOAT), transaction_type (STRING, values are 'Credit' or 'Debit'), counterparty_name (STRING), counterparty_account (STRING).
        """

    def _execute_query(self, sql_query: str) -> Tuple[str, 'pd.DataFrame']:
        """
        Execute SQL and return both a text summary and structured data (DataFrame).
        """
        try:
            clean_sql = sql_query.strip().replace("```sql", "").replace("```", "")
            print(f"--- Executing SQL: ---\n{clean_sql}\n--------------------")
            query_job = self.bq_client.query(clean_sql)
            results = query_job.to_dataframe()
            if results.empty:
                return "The query executed successfully but returned no results.", None

        # Return BOTH textual representation and DataFrame for the UI
            return results.to_string(index=False), results
        except Exception as e:
            return f"An error occurred while executing the BigQuery query: {e}", None


    def generate_sql(self, state: GraphState) -> Dict[str, Any]:
        question = state["messages"][-1].content
        chat_history = state["messages"]

        # Format prior messages for follow-up queries
        formatted_history = []
        for msg in chat_history[:-1]:
            if isinstance(msg, HumanMessage):
                formatted_history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_history.append(f"Assistant: {msg.content}")
        history_str = "\n".join(formatted_history)

        sql_prompt = ChatPromptTemplate.from_messages([
            ("system",
            f"""
            You are a BigQuery SQL expert. Given a user question and the database schema, write a valid **BigQuery** SQL query to answer it.
            Pay close attention to the conversation history to maintain filters for follow-up questions.
            Only output the SQL query and nothing else. Ensure tables are referenced as `{BIGQUERY_DATASET}.table_name`.
            Database schema:
            {self.schema}

            Conversation history:
            {history_str}
            """),
            ("human", "{question}")
        ])
        
        # Generate SQL query using the prompt and LLM
        chain = sql_prompt | self.llm
        response = chain.invoke({"question": question})
        return {"generated_sql": response.content, "messages": chat_history}

    def run(self, query: str, conversation_history: Sequence[BaseMessage], return_full_history: bool = True) -> Any:
        """
        Process a user query and return the response.

        Args:
            query: The user's query
            conversation_history: List of previous messages in the conversation
            return_full_history: If True, return payload and full history. If False, return only the payload.

        Returns:
            If return_full_history is True: Tuple of (response_payload: Dict, updated_messages: List[BaseMessage])
            If return_full_history is False: response_payload: Dict
        """
        from langchain_core.messages import AIMessage

        messages = list(conversation_history)

        try:
            # Generate SQL query
            sql_result = self.generate_sql({"messages": messages, "sql_result": "", "generated_sql": ""})
            sql_query = sql_result["generated_sql"]

            # Execute the query to get both text and DataFrame
            text_result, df_result = self._execute_query(sql_query)

            # Create a concise summary of the result for the LLM
            if df_result is not None and not df_result.empty:
                if df_result.shape == (1, 1):
                    # Handle single-value aggregate results
                    value = df_result.iloc[0, 0]
                    result_summary_for_llm = f"The user's question was '{query}'. The query returned a single value: {value}"
                else:
                    # Handle multi-row table results
                    result_summary_for_llm = f"The query returned a table with {len(df_result)} rows. The user's question was: '{query}'"
            else:
                result_summary_for_llm = text_result  # Handles "no results" or errors

            # Always generate a conversational summary based on the text result
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a helpful data assistant. Your goal is to provide a concise, conversational answer based on the user's question and the summary of the query result.

                1.  If the result summary says 'returned a single value', answer the user's question directly with that value.
                    - Example Input: "The user's question was 'What is the average transaction amount?'. The query returned a single value: 1520.75"
                    - Example Output: "The average transaction amount is 1520.75."

                2.  If the result summary says 'returned a table with X rows', state that you found that many items.
                    - Example Input: "The query returned a table with 412 rows. The user's question was: 'show me transactions under 2000'"
                    - Example Output: "I found 412 transactions that match your criteria."

                3.  If the result is an error or has no results, state that clearly.
                """),
                ("human", "{result_summary}")
            ])
            summary_chain = summary_prompt | self.llm
            summary = summary_chain.invoke({
                "result_summary": result_summary_for_llm
            }).content

            # The message for the chat history is the summary
            messages.append(AIMessage(content=summary))

            # The return payload contains both the summary and the raw DataFrame
            response_payload = {
                "summary": summary,
                "dataframe": df_result
            }

            if return_full_history:
                return response_payload, messages
            else:
                return response_payload

        except Exception as e:
            error_msg = f"I encountered an error while processing your query: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            if return_full_history:
                # The first element is the summary, the second is the dataframe
                return {"summary": error_msg, "dataframe": None}, messages
            else:
                return {"summary": error_msg, "dataframe": None}
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your query: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            return error_msg, messages