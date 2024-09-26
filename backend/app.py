from flask import Flask, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import time
from dotenv import load_dotenv


app = Flask(__name__)

load_dotenv()

groq_api_key = "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"
cohere_api_key = "T61weFNkwPHnPpw5EUEFvPA7jyEYbdvTn4bupvkD"

# Initialize components
embeddings = CohereEmbeddings(
    cohere_api_key=cohere_api_key,
    model="embed-english-v3.0"  # You can change this to another appropriate model
)
loader = PyPDFLoader("files/result.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs[:50])
vectors = FAISS.from_documents(final_documents, embeddings)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    Please provide the most accurate response based on the provided question.
    There is no need to say "based on the context".
    Just think you are talking to a very valuable client.
    Be polite.
    """),
    ("human", "{context}"),
    ("human", "{question}")
])

retriever = vectors.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input')
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    start = time.process_time()
    response = qa_chain({"query": user_input})
    process_time = time.process_time() - start

    output = response['result']

    return jsonify({
        "question": user_input,
        "answer": output,
        "process_time": process_time
    })

if __name__ == '__main__':
    app.run(debug=True)