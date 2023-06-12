#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Androse_Jes
#
# Created:     28-05-2023
# Copyright:   (c) Androse_Jes 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os, re
import streamlit as st
from langchain import OpenAI
#from apikey import openai_apikey
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.set_page_config(page_title="Living NPC", page_icon=":zany_face:")

st.markdown("# Living NPC 👻")
st.sidebar.markdown("# Living NPC 👻")


st.write("""Experience LIVING NPC: Chat with YouTube and PDF files seamlessly.
Engage in real-time conversations with videos and extract valuable data from PDFs.
Explore a new dimension of interactive browsing today.""")

#os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]

#embedding
#instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})

@st.cache_resource
def embedding():
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})
    return instructor_embeddings
