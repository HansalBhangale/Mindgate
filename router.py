import config  # Ensures auth is set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def get_router_chain():
    """Initializes the classification chain to route user queries."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=config.OPENAI_API_KEY)

    prompt_template = """Your job is to classify a user's question into one of two categories based on its content: 'BIGQUERY' or 'UPI_PDF'.
Do not answer the question. Your only output should be the category name.

- 'BIGQUERY': Choose this for questions about specific customers, bank transactions, accounts, or any financial calculations (like total, average, count, list, who, what) that require looking up data in a database.
  Examples: "List the last 5 transactions for customer 88.", "What is the total transaction amount for credits?", "Show me the details for Carmen Strosin."

- 'UPI_PDF': Choose this for general knowledge questions about the Unified Payments Interface (UPI). This includes questions about how it works, its features, security, transaction limits, or its history.
  Examples: "What is a VPA in UPI?", "How do QR code payments function?", "Explain the security model of UPI.", "What are the daily transaction limits?"

Question: {question}
Category:"""

    prompt = PromptTemplate.from_template(prompt_template)
    return prompt | llm

def route_query(chain, query: str) -> str:
    """Uses the chain to classify the query and returns the destination."""
    result = chain.invoke({"question": query})
    decision = result.content.strip()
    print(f"--- Routing decision: {decision} ---")
    if "BIGQUERY" in decision:
        return "BIGQUERY"
    return "UPI_PDF"