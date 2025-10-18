from typing import Dict, Any, TypedDict, Sequence, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from config import OPENAI_API_KEY


class RAGState(TypedDict):
    messages: Sequence[BaseMessage]
    context: str
    question: str


class PDFRAGChain:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vector_store = self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the FAISS vector store from disk."""
        from langchain_community.vectorstores import FAISS
        try:
            return FAISS.load_local("vector_store_openai", self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Please run 'python pdf_indexer.py' first to create the vector store.")
            raise

    def rephrase_question(self, state: RAGState) -> Dict[str, Any]:
        """Rephrase the user's question to be a standalone question based on chat history."""
        if not state["messages"]:
            return {"question": "", "messages": state["messages"]}

        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            return {"question": "", "messages": state["messages"]}

        chat_history = state["messages"][:-1]
        
        if not chat_history:
            return {"question": last_message.content, "messages": state["messages"]}

        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = rephrase_prompt | self.llm
        
        response = chain.invoke({
            "chat_history": chat_history,
            "question": last_message.content
        })
        
        return {"question": response.content, "messages": state["messages"], "context": state.get("context", "")}

    def retrieve_context(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve relevant context from the vector store."""
        if not state.get("question"):
            return {"context": "", "messages": state["messages"], "question": state.get("question", "")}
            
        # Search for relevant context using the rephrased question
        query = state["question"]
        docs = self.vector_store.similarity_search(query, k=3)
        
        # Format the context
        context = "\n\n".join([f"Context {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        return {"context": context, "messages": state["messages"], "question": state["question"]}

    def generate_response(self, state: RAGState) -> Dict[str, Any]:
        if not state["messages"] or not isinstance(state["messages"][-1], HumanMessage):
            return {"messages": state["messages"]}
            
        question = state["messages"][-1].content
        context = state["context"]

        if not context.strip():
            # Add a check to see if the question is about UPI
            # This is a simple check, can be improved with another LLM call if needed
            if "upi" not in question.lower():
                 return {
                    "messages": [
                        *state["messages"],
                        AIMessage(content="I can only answer questions about UPI based on the provided document. Please ask a question about UPI.")
                    ]
                }
            return {
                "messages": [
                    *state["messages"],
                    AIMessage(content="I couldn't find any relevant information in the document to answer your question about UPI.")
                ]
            }

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            """
            You are a helpful assistant that answers questions about UPI (Unified Payments Interface) and related topics.
            Use the provided context to answer the user's question. The context comes from relevant sections of a document.
            Your answer should be based SOLELY on the context provided. Do not use any prior knowledge.
            
            Guidelines:
            - Be accurate and concise in your responses.
            - If the context does not contain the answer, say that you cannot find the information in the document.
            - If the question is not about UPI or related topics, politely inform the user.
            - Use markdown to format your response when appropriate.
            
            Context:
            {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # Prepare chat history
        chat_history = state["messages"][:-1]
        
        # Create and invoke the chain
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history
        })
        
        return {"messages": [*state["messages"], response]}
        
    def run(self, query: str, conversation_history: Sequence[BaseMessage]) -> Tuple[str, Sequence[BaseMessage]]:
        """
        Process a user query about UPI using the PDF RAG system.
        
        Args:
            query: The user's query about UPI (Note: this is for compatibility, the actual message is the last one in conversation_history)
            conversation_history: List of messages in the conversation, including the latest user query.
            
        Returns:
            Tuple of (answer: str, updated_messages: List[BaseMessage])
        """
        # The conversation_history from app.py already contains the new user query.
        state = {
            "messages": list(conversation_history),
            "context": "",
            "question": ""
        }
        
        # Process the query through the workflow
        result = self.workflow.invoke(state)
        
        # Extract the last message (assistant's response)
        answer = result["messages"][-1].content
        
        return answer, result["messages"]
        
    @property
    def workflow(self):
        """Lazily initialize the LangGraph workflow."""
        if not hasattr(self, '_workflow'):
            self._workflow = StateGraph(RAGState)
            self._workflow.add_node("rephrase", self.rephrase_question)
            self._workflow.add_node("retrieve", self.retrieve_context)
            self._workflow.add_node("generate", self.generate_response)
            self._workflow.set_entry_point("rephrase")
            self._workflow.add_edge("rephrase", "retrieve")
            self._workflow.add_edge("retrieve", "generate")
            self._workflow.add_edge("generate", END)
            self._workflow = self._workflow.compile()
        return self._workflow
