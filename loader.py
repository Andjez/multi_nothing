#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      Androse_Jes
#
# Created:     12-06-2023
# Copyright:   (c) Androse_Jes 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
import pikepdf
from PyPDF2 import PdfReader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader

def pdf_loader(uploaded_files):
    raw_text = ''
    for file in uploaded_files:
        bytes_data = file.read()
        with pikepdf.open(bytes_data) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text
    text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
     )
    texts = text_splitter.split_text(raw_text)
    return texts


def youtube_loader(yt_link):
    loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts
