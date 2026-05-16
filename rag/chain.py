import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions strictly based on the provided document context.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer using ONLY the information in the context above.
- If the answer is not in the context, say "I couldn't find that in the document."
- Be concise and clear.

Answer:
"""


def build_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def ask_question(qa, question: str) -> dict:
    chain = qa["chain"]
    retriever = qa["retriever"]

    answer = chain.invoke(question)

    source_docs = retriever.invoke(question)
    sources = []
    for doc in source_docs:
        sources.append({
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content[:200].strip() + "..."
        })

    return {"answer": answer, "sources": sources}