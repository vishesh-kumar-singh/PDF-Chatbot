import numpy as np
import pymupdf
import re
from sentence_transformers import SentenceTransformer
import torch
from rank_bm25 import BM25Okapi

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    text = re.sub(r'(?<!\.)\n', ' ', text)
    return text


def get_chunks(pdf):
    file=pymupdf.open(stream=pdf.read(), filetype="pdf")
    total_text=""
    for page in file:
        text=page.get_text()

        total_text+=" "+clean_text(text)
    paragraphs=total_text.split('\n')
    return paragraphs

def embed(paragraphs):

    chunk_embeddings = embedding_model.encode(paragraphs, convert_to_numpy=True,normalize_embeddings=True,device=device)
    return chunk_embeddings

def top_7_results(query,chunk_embeddings):
    query_embedded= embedding_model.encode(query, convert_to_numpy=True,normalize_embeddings=True)

    simmilarities=np.dot(chunk_embeddings,query_embedded)
    top_7_idx=np.argsort(simmilarities,axis=0)[-7:][::-1].tolist()
    return top_7_idx

# Reciprocal Rank Fusion Function
def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)  # RRF formula

    # Sort by fused scores (higher is better)
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]


def refine_results(query,paragraphs,ids):

    filtered_paragraphs=[]
    for id in ids:
        filtered_paragraphs.append(paragraphs[id])
    # Create BM25 object from the filtered paragraphs
    tokenized_paragraphs = [paragraph.split() for paragraph in filtered_paragraphs]
    bm25 = BM25Okapi(tokenized_paragraphs)

    # Get BM25 scores for the query
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Rank documents based on BM25 (higher score = better rank)
    bm25_ranking = np.argsort(bm25_scores)[::-1]  # Sort in descending order

    # Get semantic search ranking (assumed precomputed)
    semantic_ranking = list(range(len(paragraphs)))  # Example: Replace with actual ranking

    # Apply RRF on BM25 + Semantic Search rankings
    fused_ranking = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])

    # Get top 3 refined results
    top_3_results = [filtered_paragraphs[i] for i in fused_ranking[:3]]

    return top_3_results



import streamlit as st



uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])



if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None
    st.session_state.chunk_embeddings = None
    st.session_state.switch=1


if uploaded_file and st.session_state.switch==1:
    with st.spinner("Processing PDF..."):
        # Extract text
        st.session_state.text_chunks = get_chunks(uploaded_file)
        st.session_state.chunk_embeddings = embed(st.session_state.text_chunks)
    
    st.success("✅ PDF Processed Successfully!")
    st.session_state.switch=0


query = st.text_input("Ask something from the document:")

if query and st.session_state.chunk_embeddings is None:
    st.error("Please upload a PDF file first")

if query and st.session_state.text_chunks is not None:

    with st.spinner("Searching..."):
        ids = top_7_results(query, st.session_state.chunk_embeddings)
    with st.spinner("Refining..."):
        results=refine_results(query,st.session_state.text_chunks,ids)
    with st.spinner("Generating Answer..."):
        ans1=results[0]
        ans2=results[1]
        ans3=results[2]
    st.header("These are the three expected answers for your query!")
    st.subheader("Expected Answer 1:")
    st.write(ans1)
    st.subheader("Expected Answer 2:")
    st.write(ans2)
    st.subheader("Expected Answer 3:")
    st.write(ans3)

if not uploaded_file:
    st.session_state.text_chunks = None
    st.session_state.chunk_embeddings = None
    st.session_state.switch=1