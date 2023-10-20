from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from flask import Flask, request
from flask_cors import CORS

import sys

app = Flask(__name__)
CORS(app)

data = None
vectorstore = None
qa = None

def load_data():
    urls = ["https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-banking-and-finance/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-behavioral-design/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-financial-management/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-general-management/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-marketing/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-politics-and-management/",
        "https://www.zhaw.ch/de/sml/studium/bachelor/betriebsoekonomie-risk-and-insurance/"]
    loader = WebBaseLoader(urls)
    global data
    data = loader.load()

def vectorize_data():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    api_key = "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"
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