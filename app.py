from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from ressources import config
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from flask import Flask, request
from flask_cors import CORS

from bs4 import BeautifulSoup
import sys
import json

app = Flask(__name__)
CORS(app)

data = None
vectorstore = None
qa = None

def load_data():
    with open('ressources/sources.json', 'r') as config_file:
        sources = json.load(config_file)
    # Extract the 'urls' list from the configuration
    urls = sources.get('urls', [])
    loader = WebBaseLoader(urls)
    
    global data
    data = loader.load()
    for doc in data:
        # Remove the html tags from the documents
        cleaned_text = clean_document(doc)
    doc.page_content = cleaned_text

def clean_document(document):
    page_content = document.page_content
    soup = BeautifulSoup(page_content, 'html.parser')
    clean_text = ' '.join(soup.get_text().split())
    return clean_text

def vectorize_data():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    api_key = config.OPENAI_API_KEY
    openai_embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    global vectorstore
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=openai_embeddings)


def init_retriever():
    llm = OpenAI(temperature=0)
    memory = ConversationSummaryMemory(llm=llm)
    memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)


    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    global qa
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


@app.route("/")
def landing():
    return "landingpage"

@app.route("/data")
def print_data():
    print(request.args)
    return str(data)


@app.route("/question", methods=['GET'])
def get_answer_by_question():
    data = request.get_json()
    question = data.get('question')
    answer = qa(question)
    return str(answer)
    
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__" or __name__ == "app" or __name__ == "flask_app":
    load_data()
    vectorize_data()
    init_retriever()
    print(sys.executable)
    print('running')