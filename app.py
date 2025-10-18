import os
import config
from router import get_router_chain, route_query
from bigquery_chain import BigQueryChain
from pdf_rag_chain import PDFRAGChain
from langchain_core.messages import HumanMessage

def main():
    print("Initializing chatbot components...")

    if not os.path.exists(config.SERVICE_ACCOUNT_PATH or ""):
        print(f"FATAL ERROR: Service account file not found at: {config.SERVICE_ACCOUNT_PATH}")
        print("Please check the GOOGLE_APPLICATION_CREDENTIALS path in your .env file.")
        return

    # Initialize chains/assistants
    router_chain = get_router_chain()
    bq_assistant = BigQueryChain()
    pdf_assistant = PDFRAGChain()

    print("\n✅ Chatbot is ready!")
    print("--- Welcome to the Hybrid Data Chatbot ---")
    print("I can answer questions about customer data from BigQuery or about UPI from a PDF.")
    print("Type 'exit' to quit.")

    # Unified conversation memory (LangGraph-friendly: list[BaseMessage])
    conversation_history = []

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Track user message in shared memory
        conversation_history.append(HumanMessage(content=user_query))

        try:
            destination = route_query(router_chain, user_query)
            if destination == "BIGQUERY":
                answer, updated_messages = bq_assistant.run(user_query, conversation_history)
            else:
                answer, updated_messages = pdf_assistant.run(user_query, conversation_history)

            # Update conversation history with the latest messages
            conversation_history = updated_messages
            print(f"\nBot: {answer}")
            
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try your query again.")

if __name__ == "__main__":
    main()