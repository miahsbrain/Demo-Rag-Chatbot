from torch import float32
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import os
import streamlit as st
import base64
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


# Loading the model
checkpoint = Path('./LaMini-T5-738M')
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto', torch_dtype=float32)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# ChromaDB setup
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
persist_directory = 'db'
vector_store = Chroma(
    collection_name='docuents',
    embedding_function=embeddings,
    persist_directory=persist_directory
)

# Data ingestation
docs = Path('./docs')

@st.cache_resource
def ingest():
    if not docs.exists():
        docs.mkdir(exist_ok=True, parents=True)
    loader = None

    for _, dir, files in os.walk(docs):
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                filepath = docs / file
                loader = PDFMinerLoader(filepath)

    if loader is not None:
        # Load and split documents into chunks
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
        texts = text_splitter.split_documents(documents=documents)
        # Create embeddings to store documents in chromadb
        vector_store.add_documents(documents=texts, embedding=embeddings)
        
    else:
        print('No sitable file exists in directory')

# Pipeline
@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        task='text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        device_map='auto',
        # temperature=.3
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def qa_pipeline():
    llm = llm_pipeline()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain)

    return qa_chain

# Display PDF
def displayPDF(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Display PDF in HTML
    html_markdown = f'<iframe src="data:application/pdf; base64, {base64_pdf}" style="width: 100%; height: 450px"></iframe>'
    return html_markdown

# Process input texts
def process_input(question):
    qa_pipe = qa_pipeline()
    result = qa_pipe.invoke({'input': f'{question}'})
    return result

# Find file size
def get_file_size(file):
    file.seek(0, os.SEEK_END)
    filesize = file.tell()
    file.seek(0)
    return filesize

st.set_page_config(page_title='Chat with your PDF', layout='wide')
def main():
    st.markdown('<h1 style="text-align: center;">Chat with your PDF</h1>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('file', type=['pdf'], label_visibility='collapsed')
    if uploaded_file is not None:
        file_details = {
            'filename': uploaded_file.name,
            'filesize': f'{get_file_size(uploaded_file)} bytes'
        }
        # Save uploaded file
        filepath = docs / uploaded_file.name
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())        

        col1, col2 = st.columns(spec=[1,2])

        with col1:
            st.markdown('<h4>File details</h4>', unsafe_allow_html=True)
            st.json(file_details)
            st.markdown('<h4>File preview</h4>', unsafe_allow_html=True)
            st.markdown(displayPDF(filepath), unsafe_allow_html=True)

        with col2:
            with st.spinner('Processing file...'):
                ingest()
            st.success('Processing complete')
            st.markdown('<h4>Chat here</h4>', unsafe_allow_html=True)


            # Chat
            user_input = st.chat_input('Ask anything...')

            if user_input:
                # Create storage
                if 'messages' not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(name=message.get('role')):
                        st.write(message.get('content'))

                st.session_state.messages.append({'role': 'user', 'content': user_input})

                # Display typed message
                with st.chat_message(name='user'):
                    st.write(user_input)

                reply = process_input(user_input)['answer']
                st.session_state.messages.append({'role': 'assistant', 'content': reply})

                # Display reply
                with st.chat_message(name='assistant'):
                    st.write(reply)


            



if __name__ == '__main__':
    main()

