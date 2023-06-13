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
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader

def pdf_loader(uploaded_files):
    pdf_text = {}
    for file in uploaded_files:
        bytes_data = file.read()
        pdf = pikepdf.open(file)
        pdf.save(file.name)
        filename = file.name
        loader = UnstructuredPDFLoader(filename)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        pdf_text.update(texts)
    return pdf_text

def youtube_loader(yt_link):
    loader = YoutubeLoader.from_youtube_url(yt_link, add_video_info=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts
