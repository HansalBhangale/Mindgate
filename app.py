import config  # This MUST be the first import to load .env variables
from router import get_router_chain, route_query
from bigquery_chain import get_bigquery_chain
from pdf_rag_chain import get_pdf_rag_chain
import os

def main():
    """
    Initializes all chatbot components and starts the main interaction loop.
    """
    print("Initializing chatbot components...")
    
    # Check if the service account file exists
    if not os.path.exists(config.SERVICE_ACCOUNT_PATH):
        print(f"FATAL ERROR: Service account file not found at: {config.SERVICE_ACCOUNT_PATH}")
        print("Please check the GOOGLE_APPLICATION_CREDENTIALS path in your .env file.")
        return

    # Initialize all the necessary chains
    router_chain = get_router_chain()
    bq_chain = get_bigquery_chain()
    pdf_chain = get_pdf_rag_chain()
    
    if not pdf_chain:
        print("\nExiting application due to an error in PDF chain initialization.")
        return

    print("\n✅ Chatbot is ready!")
    print("--- Welcome to the Hybrid Data Chatbot ---")
    print("I can answer questions about customer data from BigQuery or about UPI from a PDF.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            # 1. Route the user's query to the correct data source
            destination = route_query(router_chain, user_query)

            # 2. Execute the appropriate chain to get the answer
            if destination == "BIGQUERY":
                answer = bq_chain(user_query)
            else:  # UPI_PDF
                answer = pdf_chain(user_query)

            print(f"\nBot: {answer}")
        
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try your query again.")

if __name__ == "__main__":
    main()