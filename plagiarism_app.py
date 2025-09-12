import os
import pdfplumber
from docx import Document
import nltk
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import difflib
import tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

st.set_page_config(page_title="🔍 Student Assignment Similarity Checker", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e1e5e9; }
.risk-high { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ffaa00; font-weight: bold; }
.risk-low { color: #00cc44; font-weight: bold; }
mark { background-color: #ff9999 !important; color: black; padding: 0 2px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Student Assignment Similarity Checker")
st.markdown("*Comprehensive Academic Integrity Analysis Tool*")
st.markdown("---")

st.sidebar.header("📁 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    accept_multiple_files=True,
    type=['txt','docx','pdf','py','java','c','cpp','js']
)
use_default = st.sidebar.checkbox("Use default folder instead", value=True)
folder_path = st.sidebar.text_input("Folder path:", "D:\techbis")

def extract_text(file_path=None, uploaded_file=None):
    if uploaded_file:
        raw = uploaded_file.read()
        try: return raw.decode('utf-8')
        except: return str(raw)
    ext = os.path.splitext(file_path)[1].lower()
    if ext=='.docx':
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    if ext=='.pdf':
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
        return text
    if ext in ['.txt','.java','.c','.cpp','.js']:
        with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
            return f.read()
    return None

def extract_code_tokens(file_path):
    tokens = []
    try:
        with open(file_path, 'rb') as f:
            for toknum, tokval, *_ in tokenize.tokenize(f.readline):
                if toknum in (tokenize.ENCODING, tokenize.NEWLINE,
                              tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                    continue
                tokens.append((toknum, tokval))
    except:
        pass
    return tokens

def preprocess_text(text):
    txt = text.lower()
    txt = " ".join(txt.split())
    return sent_tokenize(txt)

def calculate_originality_scores(files, sim_matrix):
    scores = {}
    n = len(files)
    for i, f in enumerate(files):
        sims = [sim_matrix[i][j] for j in range(n) if j != i]
        max_sim = max(sims) if sims else 0
        orig = max(0, 100 - max_sim * 100)
        risk = "LOW" if orig >= 85 else "MEDIUM" if orig >= 70 else "HIGH"
        scores[f] = {
            'originality_score': round(orig,2),
            'max_similarity': round(max_sim,4),
            'risk_level': risk
        }
    return scores

def assign_grade(score):
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"

def combined_grade(tf, sem):
    return assign_grade((tf + sem) / 2)

def code_similarity(tokens_a, tokens_b):
    seq_a = [tok for _, tok in tokens_a]
    seq_b = [tok for _, tok in tokens_b]
    return difflib.SequenceMatcher(None, seq_a, seq_b).ratio()

def get_similar_sentences(fname, all_sents, sent_map, sim_matrix, threshold=0.3):
    idxs = [i for i, f in enumerate(sent_map) if f == fname]
    highlights = set()
    for i in idxs:
        for j in range(len(sent_map)):
            if sent_map[j] != fname and sim_matrix[i][j] > threshold:
                highlights.add(i)
                break
    return highlights

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def compute_tfidf(docs):
    tf = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    return tf.fit_transform(docs)

def process_assignments():
    all_sub, all_code = {}, {}
    if use_default and os.path.exists(folder_path):
        for fn in os.listdir(folder_path):
            ext = os.path.splitext(fn)[1].lower()
            fpath = os.path.join(folder_path, fn)
            if ext == '.py':
                all_code[fn] = extract_code_tokens(fpath)
            else:
                txt = extract_text(file_path=fpath)
                if txt: all_sub[fn] = preprocess_text(txt)
    elif uploaded_files:
        for uf in uploaded_files:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext == '.py':
                temp = f"temp_{uf.name}"
                with open(temp, 'wb') as f: f.write(uf.getvalue())
                all_code[uf.name] = extract_code_tokens(temp)
                os.remove(temp)
            else:
                txt = extract_text(uploaded_file=uf)
                if txt: all_sub[uf.name] = preprocess_text(txt)

    files = list(all_sub.keys())
    code_files = list(all_code.keys())

    all_sents, sent_map = [], []
    for f in files:
        for s in all_sub[f]:
            all_sents.append(s)
            sent_map.append(f)

    sent_sim = None
    if all_sents:
        vs = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        sent_sim = cosine_similarity(vs.fit_transform(all_sents))

    docs = [" ".join(all_sub[f]) for f in files]
    tfidf_sim = cosine_similarity(compute_tfidf(docs)) if docs else None

    model = load_model()
    sem_mat = None
    if docs:
        embs = model.encode(docs, convert_to_tensor=True)
        sem_mat = util.cos_sim(embs, embs).cpu().numpy()

    tfidf_scores = calculate_originality_scores(files, tfidf_sim) if tfidf_sim is not None else {}
    sem_scores = calculate_originality_scores(files, sem_mat) if sem_mat is not None else {}
    for f in files:
        tfidf_scores[f]['grade'] = assign_grade(tfidf_scores[f]['originality_score'])
        sem_scores[f]['grade_sem'] = assign_grade(sem_scores[f]['originality_score'])
        tfidf_scores[f]['combined_grade'] = combined_grade(tfidf_scores[f]['originality_score'], sem_scores[f]['originality_score'])

    st.sidebar.markdown("**Grade Distribution (Combined)**")
    if files:
        gc = pd.Series([d['combined_grade'] for d in tfidf_scores.values()]).value_counts().to_dict()
        for g in ["A","B","C","D","F"]:
            st.sidebar.write(f"{g}: {gc.get(g,0)}")
    else:
        st.sidebar.info("No text files uploaded")

    if code_files:
        st.sidebar.markdown("**Code Files:**")
        for cf in code_files: st.sidebar.write(cf)
    else:
        st.sidebar.info("No code files uploaded")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", "📊 Graph", "💻 Code", "📈 Details & Highlights", "📄 View Text with Highlights"
    ])

    with tab1:
        if files:
            df = pd.DataFrame([{
                "Assignment": f,
                "TF-IDF Orig": f"{tfidf_scores[f]['originality_score']:.1f}%",
                "Semantic Orig": f"{sem_scores[f]['originality_score']:.1f}%",
                "Combined Grade": tfidf_scores[f]['combined_grade'],
                "TF-IDF Risk": tfidf_scores[f]['risk_level'],
                "Semantic Risk": sem_scores[f]['risk_level'],
                "Max Sim": round(max(tfidf_scores[f]['max_similarity'], sem_scores[f]['max_similarity']),4)
            } for f in files])
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv")
        else:
            st.info("No text submissions")

    with tab2:
        if files:
            st.subheader("Assignment vs Max Similarity")
            bar_df = pd.DataFrame({
                "Assignment": files,
                "Max Similarity": [tfidf_scores[f]['max_similarity'] for f in files]
            })
            fig = go.Figure(go.Bar(x=bar_df["Assignment"], y=bar_df["Max Similarity"], marker_color='indianred'))
            fig.update_layout(title="Assignment vs Max Similarity", xaxis_title="Assignment", yaxis_title="Max Similarity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No text submissions")

    with tab3:
        if code_files:
            sim_mat = np.zeros((len(code_files), len(code_files)))
            for i,a in enumerate(code_files):
                for j,b in enumerate(code_files):
                    if i!=j:
                        sim_mat[i,j] = code_similarity(all_code[a], all_code[b])
            df_code = pd.DataFrame(sim_mat, index=code_files, columns=code_files)
            st.subheader("Code Similarity Matrix")
            st.dataframe(df_code.style.background_gradient(cmap='Reds'), use_container_width=True)
        else:
            st.info("No code submissions")

    with tab4:
        if files:
            sel = st.selectbox("Select file", files)
            idx = files.index(sel)
            st.subheader("TF-IDF Top Matches")
            for fn, sc in sorted([(files[j], tfidf_sim[idx][j]) for j in range(len(files)) if j!=idx], key=lambda x: -x[1])[:5]:
                st.write(f"{fn}: {sc:.4f}")
            st.subheader("Semantic Top Matches")
            for fn, sc in sorted([(files[j], sem_mat[idx][j]) for j in range(len(files)) if j!=idx], key=lambda x: -x[1])[:5]:
                st.write(f"{fn}: {sc:.4f}")
            highlights = get_similar_sentences(sel, all_sents, sent_map, sent_sim)
            st.subheader(f"Highlighted Sentences in {sel}")
            disp = []
            for i,s in enumerate(all_sents):
                if sent_map[i]==sel:
                    disp.append(f"<mark>{s}</mark>" if i in highlights else s)
            st.markdown(f"<div style='border:1px solid #ddd; padding:10px; max-height:300px; overflow-y:auto;'>{' '.join(disp)}</div>", unsafe_allow_html=True)
        else:
            st.info("No text submissions")

    with tab5:
        if files:
            st.header("📄 View All Assignments with Highlighted Similar Sentences")
            for fname in files:
                highlights = get_similar_sentences(fname, all_sents, sent_map, sent_sim)
                st.subheader(fname)
                disp=[]
                for i,s in enumerate(all_sents):
                    if sent_map[i]==fname:
                        disp.append(f"<mark>{s}</mark>" if i in highlights else s)
                st.markdown(f"<div style='border:1px solid #ddd; padding:10px; margin-bottom:20px; max-height:200px; overflow-y:auto;'>{' '.join(disp)}</div>", unsafe_allow_html=True)
        else:
            st.info("No text submissions")

if __name__ == "__main__":
    if st.button("🚀 Start Analysis"):
        process_assignments()
    st.markdown("---")
    st.markdown("*Hackathon Academic Integrity Tool*")
