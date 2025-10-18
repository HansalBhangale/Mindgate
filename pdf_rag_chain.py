import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from config import VECTOR_STORE_PATH_OPENAI, OPENAI_API_KEY

# ... (rest of the file remains the same)
def get_pdf_rag_chain():
    """
    Initializes the RAG chain for answering questions from the PDF
    using OpenAI models.
    """
    print("--- Loading OpenAI vector store for PDF... ---")
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH_OPENAI, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"FATAL: Failed to load vector store. Did you run 'python pdf_indexer.py' first? Error: {e}")
        return None

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """
    Answer the user's question based *only* on the context provided below.
    If the information is not in the context, state clearly "I could not find that information in the provided document."
    Be concise and directly address the question.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    rag_chain = prompt | llm

    def run_full_chain(question: str) -> str:
        print("--- Retrieving relevant info from PDF... ---")
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        print("--- Generating answer with OpenAI... ---")
        return rag_chain.invoke({"context": context, "question": question}).content

    return run_full_chain