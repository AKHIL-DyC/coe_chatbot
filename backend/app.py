from flask import Flask, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

groq_api_key = "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"
cohere_api_key = "T61weFNkwPHnPpw5EUEFvPA7jyEYbdvTn4bupvkD"

# Initialize components
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
loader = PyPDFLoader("files/result.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
vectors = FAISS.from_documents(final_documents, embeddings)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    Please provide the most accurate response based on the provided question.
    There is no need to say "based on the context".
    Just think you are talking to a very valuable client.
    Be polite.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    process_time = time.process_time() - start

    output = response['answer']

    return jsonify({
        "question": user_input,
        "answer": output,
        "process_time": process_time
    })

if __name__ == '__main__':
    app.run(debug=True)