from typing import Dict, Any, TypedDict, Annotated, Sequence, Tuple
from google.cloud import bigquery
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from config import GCP_PROJECT_ID, BIGQUERY_DATASET, OPENAI_API_KEY




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

    def _execute_query(self, sql_query: str) -> str:
        try:
            clean_sql = sql_query.strip().replace("```sql", "").replace("```", "")
            print(f"--- Executing SQL: ---\n{clean_sql}\n--------------------")
            query_job = self.bq_client.query(clean_sql)
            results = query_job.to_dataframe()
            if results.empty:
                return "The query executed successfully but returned no results."
            return results.to_string()
        except Exception as e:
            return f"An error occurred while executing the BigQuery query: {e}"

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

    def run(self, query: str, conversation_history: Sequence[BaseMessage]) -> Tuple[str, Sequence[BaseMessage]]:
        """
        Process a user query and return the response along with updated conversation history.
        
        Args:
            query: The user's query
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Tuple of (answer: str, updated_messages: List[BaseMessage])
        """
        from langchain_core.messages import AIMessage
        
        # Add the user's message to the conversation history
        messages = list(conversation_history)
        
        try:
            # Generate SQL query
            sql_result = self.generate_sql({"messages": messages, "sql_result": "", "generated_sql": ""})
            sql_query = sql_result["generated_sql"]
            
            # Execute the query
            query_result = self._execute_query(sql_query)
            
            # Format the response
            # ✅ Conversational summarization logic
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a helpful data assistant. Explain the database result conversationally.
                If it looks like a table, show the table in clear format.
                If it's just one value, answer directly in a single friendly sentence.
                Avoid SQL code blocks or technical formatting.
                """),
                ("human", "Question: {question}\nResult: {result}")
            ])

            summary_chain = summary_prompt | self.llm
            summary = summary_chain.invoke({
                "question": query,
                "result": query_result
            }).content

            # Final assistant message
            response = summary
            messages.append(AIMessage(content=response))            
            # Add the assistant's response to the conversation history
            messages.append(AIMessage(content=response))
            
            return response, messages
            
        except Exception as e:
            error_msg = f"I encountered an error while processing your query: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            return error_msg, messages