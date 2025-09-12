import os
import pdfplumber
from docx import Document
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# Function to extract text based on file type
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    # DOCX
    if ext == '.docx':
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    
    # PDF
    elif ext == '.pdf':
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    
    # TXT or code files
    elif ext in ['.txt', '.py', '.java', '.c', '.cpp', '.js']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    
    else:
        print(f"Unsupported file type: {file_path}")
        return ""
    
    return text

# Function to preprocess text
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove extra whitespace
    text = " ".join(text.split())
    # split into sentences
    sentences = sent_tokenize(text)
    return sentences

# Example: process all files in a folder
folder_path = "C:\\Users\\GOKULKAVIN\\plagiarism_mvp\\Text_Files"
all_submissions = {}

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    text = extract_text(file_path)
    sentences = preprocess_text(text)
    all_submissions[filename] = sentences

# Test output
for fname, sents in all_submissions.items():
    print(f"\n=== {fname} ===")
    print(sents)  # print first 5 sentences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# all_submissions: {filename: [sent1, sent2, ...]}
# Make sure all_submissions is already filled from your preprocessing step
filenames = list(all_submissions.keys())

# Flatten sentences and map them to filenames
sentences = []
sent_map = []
for fname, sents in all_submissions.items():
    for s in sents:
        sentences.append(s)
        sent_map.append(fname)

# TF-IDF on all sentences
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sentences)

# Sentence-level cosine similarity
sim_matrix = cosine_similarity(tfidf_matrix)

# Function to get top matching sentences between two files
def top_matching_sentences(file_a, file_b, top_k=3):
    idx_a = [i for i, f in enumerate(sent_map) if f == file_a]
    idx_b = [i for i, f in enumerate(sent_map) if f == file_b]
    
    scores = []
    for i in idx_a:
        for j in idx_b:
            scores.append((sim_matrix[i, j], sentences[i], sentences[j]))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# Compute document-level similarity
# Flatten sentences back to full document for TF-IDF baseline similarity
docs = [" ".join(all_submissions[fname]) for fname in filenames]
vectorizer_doc = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
tfidf_docs = vectorizer_doc.fit_transform(docs)
doc_sim_matrix = cosine_similarity(tfidf_docs)

K = 3  # Top 3 matches per submission
for i, fname in enumerate(filenames):
    doc_sim_matrix[i, i] = -1  # ignore self
    top_indices = np.argsort(doc_sim_matrix[i])[::-1][:K]
    
    print(f"\nTop {K} matches for '{fname}':")
    for idx in top_indices:
        match_file = filenames[idx]
        score = doc_sim_matrix[i, idx]
        print(f"\n  {match_file} -> Similarity Score: {score:.2f}")
        
        # Show top matching sentences
        matches = top_matching_sentences(fname, match_file, top_k=3)
        print("  Top matching sentences:")
        for s_score, s_a, s_b in matches:
            print(f"    Score: {s_score:.2f}")
            print(f"      {fname}: {s_a}")
            print(f"      {match_file}: {s_b}")


from sentence_transformers import SentenceTransformer, util

# docs: list of full submissions (flattened sentences)
docs = [" ".join(sents) for sents in all_submissions.values()]
filenames = list(all_submissions.keys())

# Load a fast pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Compute embeddings for each document
embs = model.encode(docs, convert_to_tensor=True)

# Step 2: Compute cosine similarity between embeddings
cosine_scores = util.cos_sim(embs, embs)  # NxN matrix

# Step 3: Show top 3 similar submissions for each document
import torch

K = 3
for i, fname in enumerate(filenames):
    # set self-similarity to -1 to ignore
    cosine_scores[i, i] = -1
    top_indices = torch.topk(cosine_scores[i], K).indices.tolist()
    
    print(f"\nTop {K} semantic matches for '{fname}':")
    for idx in top_indices:
        print(f"  {filenames[idx]} -> Semantic Score: {cosine_scores[i, idx]:.2f}")
