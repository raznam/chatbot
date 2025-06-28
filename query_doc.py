import os
import time
import re
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemma model through Groq
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="gemma2-9b-it"
)

# QA Prompt Function using Groq-Gemma
def qa_prompt_fn(question: str, context: str) -> str:
    prompt = f"""
You are an AI assistant focused on teaching and simplifying complex topics by first understanding the uploaded PDF. Your job is to explain—not summarize— any concept the user queries with step-by-step clarity and full detail. Your explanations must be so clear that even a beginner can build a rock-solid understanding.
Use only the information from the PDF to build the explanation first. Then, use your general knowledge and reasoning to generate relatable, real-life examples or analogies that reinforce the concept.

Explain the queried topic in a way that helps the user build complete, deep conceptual understanding — like a great teacher would do — with detailed explanations and real-world analogies.
Respond strictly in the format below:

Topic: [Name of the concept]

Detailed Explanation (Based on PDF):
[Break the concept down clearly in simple language. Use the uploaded PDF to cover all key aspects. Go step-by-step. Avoid summaries—focus on detailed teaching.]

Key Terms and Ideas:
[If any important terms, define them here clearly in plain English.]

Why It Matters:
[Explain the relevance or importance of the topic—based on what’s covered in the PDF.]

Real-Life Analogy / Example:
[Create a relatable example or analogy using your general knowledge to make the concept easy to grasp.]

Source Insight:
[Optional quote or refer briefly to relevant parts of the PDF used for the explanation.]

If a topic cannot be clearly identified, respond with:
"Unable to identify."

    Context:
    {context}

    Question: {question}

    Answer:
    """
    response = llm.invoke(prompt)
    result = response.content if response and hasattr(response, 'content') else "No response"

    # Define regex pattern here
    pattern = (
    r"Topic: .*?\n"
    r"Detailed Explanation \(Based on PDF\): .*?\n"
    r"Key Terms and Ideas: .*?\n"
    r"Why It Matters: .*?\n"
    r"Real-Life Analogy / Example: .*?\n"
    r"(Source Insight: .*)?"
)

    # Guardrail check
    #if not re.search(pattern, result, re.DOTALL) and "Unable to identify" not in result:
     #   return "Output format validation failed. No valid response."

    return result

# Function to perform RAG
def query_documents(query: str):
    # Guardrail: Input validation
    #if not query.strip() or len(query) > 500:
      #  return "Invalid query. Please provide a concise and meaningful crime description."

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vectorstore = Chroma(
        collection_name="my_docs",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    combined_context = "\n".join([doc.page_content for doc in docs])
    return qa_prompt_fn(query, combined_context)

# Example
if __name__ == "__main__":
    query = "Which BNS section applies if a person forcibly enters someone's house at night?"
    answer = query_documents(query)
    print("Answer:\n", answer)
