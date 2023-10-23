import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# Get an OpenAI API Key before continuing
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.title("Hello, I'm `FT-Bot` your M365 Fasttrack Assistant 👓 from Softchoice ")
    st.info("Enter an OpenAI API Key to continue")
    st.info("If you are not sure on how to get your OpenAI API key:")
    st.info( " 1) Please visit https://platform.openai.com/account/api-keys")
    st.info(" 2) Click on 'Create new key' and copy and save the key in a safe location")
    st.stop()

op_ai = ChatOpenAI(model="gpt-4", temperature=0.3,verbose=False)

# Load documents from local directory
loader = DirectoryLoader('./fasttrack/', glob="**/[!.]*", loader_cls=UnstructuredPDFLoader)
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20)

documents = splitter.split_documents(docs)



# Create vector embeddings and store them in a vector database
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())                                   

#Retriever
retriever = vectorstore.as_retriever(k=3, filter=None)

# Set up memory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history")

#RetrievalChain
qa = ConversationalRetrievalChain.from_llm(op_ai, verbose=False, retriever=retriever, chain_type="stuff", memory=memory)       

#StreamlitUI
#setting up the title
st.title("Hello, I'm `FT-Bot` your M365 Fasttrack Assistant 👓 from Softchoice  ")

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me about Fasttrack!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        
        response = qa.run(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
