import streamlit as st
import nltk
#nltk.download('punkt')
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from InferSent.models import InferSent
from transformers import pipeline
import PyPDF2
from PyPDF2 import PdfReader
# Load a pre-trained Question Answering model
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def extract_text_from_pdf(pdf_docs):
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
    return text.strip()

def split_text_into_chunks(text, chunk_size=50):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    print(len(chunks))
    return chunks

def load_infersent_model(model_path, glove_path):
    params_model = {
        'bsize': 64, 
        'word_emb_dim': 300, 
        'enc_lstm_dim': 2048, 
        'pool_type': 'max', 
        'dpout_model': 0.0, 
        'version': 2
    }
    model = InferSent(params_model)
    model.load_state_dict(torch.load(model_path))
    model.set_w2v_path(glove_path) 
    model.build_vocab_k_words(K=100000)  # Load top K words from GloVe vectors
    return model

def generate_infersent_embeddings(model, sentences):
    model.build_vocab(sentences, tokenize=True)
    dict_embeddings = {}
    for i, sentence in enumerate(sentences):
        print(f"Encoding sentence {i+1}/{len(sentences)}")
        embedding = model.encode([sentence], tokenize=True)
        dict_embeddings[sentence] = embedding.reshape(-1)  # Flatten the embedding
    return dict_embeddings

# Load InferSent model and GloVe vectors
model_path = "encoder/infersent2.pkl"
glove_path = "GloVe/glove.840B.300d.txt"
infersent_model = load_infersent_model(model_path, glove_path)


def retrieve_chunks(knn, model, query, chunks):
    query_embedding = model.encode([query], tokenize=True).reshape(1, -1)
    distances, indices = knn.kneighbors(query_embedding)
    return [chunks[i] for i in indices[0]]

def formulate_response(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    composite_query = f"Context: {context}\n\nQuery: {query}"
    return composite_query,context  # or use this composite_query with your LLM




st.title("Chat with PDF Files :fire:")

if 'chunks' not in st.session_state:
    st.session_state.chunks = []

if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = []
with st.sidebar:

    st.title("Upload Your PDF")
    docs=st.file_uploader("### Upload your PDF",type=['pdf'], accept_multiple_files=True)
    
    but=st.sidebar.button('Submit and Process')
    if but:
        with st.spinner("Processing..."):
           # Process PDF and generate chunks
            text = extract_text_from_pdf(docs)
            st.session_state.chunks = []
            st.session_state.chunks.append(split_text_into_chunks(text))
            # Generate embeddings for each chunk
            infersent_embeddings = generate_infersent_embeddings(infersent_model, st.session_state.chunks[0])
            # Convert dictionary of embeddings to a list of embeddings
            st.session_state.chunk_embeddings= []
            st.session_state.chunk_embeddings.append(np.array([infersent_embeddings[chunk] for chunk in st.session_state.chunks[0]]))
            st.sidebar.success("Done")
   
#st.write(len(chunk_embeddings))
st.write("""
#### Ask your Question.
 """)
query=st.text_area("Write your Question here.","write here")
but1=st.button('Submit')
but2 = st.button('Get Context for Answering')
# Example query
if 'context' not in st.session_state:
    st.session_state.context = ""
if but1:
    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knn.fit(st.session_state.chunk_embeddings[0])
    retrieved_chunks = retrieve_chunks(knn, infersent_model, query, st.session_state.chunks[0])
    response = formulate_response(query, retrieved_chunks)
    #st.write(response)

    # Define the context and query

    st.session_state.context = response[1]
    # Use the QA model to get the answer
    result = qa_model(question=query, context=st.session_state.context)

    # Print the answer
    st.write("Answer:", result['answer']) 

if but2:
    st.write(f'''Contexts for Answering (Top 5 chunks)
    
             {st.session_state.context}
             
             ''')