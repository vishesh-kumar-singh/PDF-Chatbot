# PDF Data Extraction and Search

A Streamlit-based application that allows users to upload PDF documents and perform intelligent searches by asking queries about the required information. This project demonstrates an application of concept of **RAG** combined with other traditional search methods.

## Project Challenge

The key challenge of this project was to develop an effective document search system. Instead of leveraging pre-trained models like GPT or BERT, we implemented a hybrid approach that combines:

- Semantic search using lightweight sentence transformers
- Traditional information retrieval with BM25
- Reciprocal Rank Fusion for result optimization

This approach offers several advantages:
- Lower computational requirements
- Faster processing times
- More predictable and controlled results
- Easier deployment and maintenance

The solution demonstrates that effective document search and retrieval can be achieved through careful combination of traditional and modern techniques, without the overhead of LLMs.

## How it Works

The application combines two powerful search approaches to provide accurate and relevant results:

1. **Semantic Search**
   - Uses the all-MiniLM-L6-v2 model from Sentence Transformers
   - Converts text into dense vector embeddings that capture semantic meaning
   - Finds similar content based on meaning, not just exact word matches
   - Particularly effective for conceptual queries and paraphrased content

2. **BM25 Ranking**
   - Classic information retrieval algorithm based on word frequency
   - Ranks documents based on term frequency and inverse document frequency (TF-IDF)
   - Excellent at matching specific keywords and phrases
   - Helps balance the semantic search with exact term matching

3. **Result Fusion**
   - Combines results from both approaches using Reciprocal Rank Fusion (RRF)
   - RRF gives higher weight to documents ranked well by both methods
   - Provides a balanced set of results that considers both semantic similarity and keyword relevance

4. **Processing Pipeline**
   - PDF text extraction and cleaning
   - Text chunking into manageable paragraphs
   - Scemantic search for the top 7 resluts
   - RRF over the 7 filtered results to give the best 3 results
   - Result ranking and presentation of top 3 most relevant passages

## Features

- PDF document upload and processing
- Semantic search using Sentence Transformers (all-MiniLM-L6-v2)
- BM25-based text ranking
- Reciprocal Rank Fusion (RRF) for combining search results
- Interactive web interface using Streamlit

## Demo

Watch the demo video on YouTube:

[![PDF Data Extraction Demo](https://img.youtube.com/vi/WU7s754RnfU/0.jpg)](https://youtu.be/WU7s754RnfU)



## Installation

1. Create and activate a conda environment:
```bash
conda create -n pdfbot python=3.12
conda activate pdfbot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run App.py
```

2. Upload a PDF document using the file uploader in the sidebar
3. Enter your query in the text input field
4. View the top 3 most relevant results from your document

