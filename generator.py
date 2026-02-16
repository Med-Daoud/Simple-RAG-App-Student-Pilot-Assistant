

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from config import *

def generate_answer(context_docs, question):
    model = OllamaLLM(model=LLM_MODEL)

    template = """
    You are a helpful assistant.
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    chain = prompt | model

    return chain.invoke({
        "context": context_text,
        "question": question
    })
