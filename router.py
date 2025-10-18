import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate




def get_router_chain():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=config.OPENAI_API_KEY)

    prompt_template = """
    Your job is to classify a user's question into one of two categories: 'BIGQUERY' or 'UPI_PDF'.
    Do not answer the question. Output exactly one word: BIGQUERY or UPI_PDF.

    - BIGQUERY: questions about customers/transactions/accounts/calculations that need database lookup.
    - UPI_PDF: questions about UPI (how it works, features, security, limits, history).

    Question: {question}
    Category:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt | llm


def route_query(chain, query: str) -> str:
    result = chain.invoke({"question": query})
    decision = (result.content or "").strip().upper()
    print(f"--- Routing decision: {decision} ---")
    return "BIGQUERY" if "BIGQUERY" in decision else "UPI_PDF"