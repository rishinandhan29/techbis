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

st.set_page_config(page_title="Assignment Authenticity Monitor", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e1e5e9; }
.risk-high { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ffaa00; font-weight: bold; }
.risk-low { color: #00cc44; font-weight: bold; }
mark { background-color: #ff9999 !important; color: black; padding: 0 2px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

st.title("Assignment Authenticity Monitor")
st.markdown("Comprehensive Academic Integrity Analysis Tool")
st.markdown("---")

st.sidebar.header("📁 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Choose files", accept_multiple_files=True,
    type=['txt', 'docx', 'pdf', 'py', 'java', 'c', 'cpp', 'js']
)
use_default = st.sidebar.checkbox("Use default folder instead", value=False)

def extract_text(file_path=None, uploaded_file=None):
    if uploaded_file:
        raw = uploaded_file.read()
        try:
            return raw.decode('utf-8')
        except:
            return str(raw)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.docx':
        return "\n".join(p.text for p in Document(file_path).paragraphs)
    if ext == '.pdf':
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    if ext in ['.txt', '.java', '.c', '.cpp', '.js']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
    for i,f in enumerate(files):
        sims = [sim_matrix[i][j] for j in range(n) if j != i]
        max_sim = max(sims) if sims else 0
        orig = max(0, 100 - max_sim * 100)
        risk = "LOW" if orig >= 85 else "MEDIUM" if orig >= 70 else "HIGH"
        scores[f] = {
            'originality_score': round(orig, 2),
            'max_similarity': round(max_sim, 4),
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

def code_grade(score):
    if score >= 0.85:
        return "F"
    elif score >= 0.60:
        return "D"
    elif score >= 0.40:
        return "C"
    elif score >= 0.20:
        return "B"
    else:
        return "A"

def code_similarity(a, b):
    seq_a = [tok for _, tok in a]
    seq_b = [tok for _, tok in b]
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

def generate_feedback_text(assignment_name, tfidf_risk, sem_risk):
    feedback_parts = []

    if tfidf_risk == "HIGH":
        feedback_parts.append(
            "Your assignment shows a *high level of direct textual similarity* with other submissions. "
            "To reduce this, rewrite the content using your own words and avoid copy-pasting large blocks."
        )
    elif tfidf_risk == "MEDIUM":
        feedback_parts.append(
            "There is a moderate amount of similar text detected. Try to paraphrase sentences and structure "
            "your content uniquely."
        )
    else:
        feedback_parts.append(
            "Textual similarity is low. Keep maintaining originality by writing in your own style."
        )

    if sem_risk == "HIGH":
        feedback_parts.append(
            "Semantic similarity indicates your assignment closely matches others in idea or meaning. "
            "Incorporate your own understanding, perspectives, and analysis to make your work distinctive."
        )
    elif sem_risk == "MEDIUM":
        feedback_parts.append(
            "There is moderate semantic overlap. Focus on adding unique insights and personal reflections."
        )
    else:
        feedback_parts.append(
            "Semantic uniqueness is good. Continue to build on original ideas."
        )

    return f"### Feedback for '{assignment_name}' (Text Files)\n\n" + "\n\n".join(feedback_parts)

def generate_feedback_code(assignment_name, risk_level):
    if risk_level == "HIGH":
        feedback = f"""
        ### Feedback for '{assignment_name}' (Code File)

        Your code shows *high similarity* to other submissions according to token-based analysis. To improve originality:
        - Try changing variable names and code structure.
        - Refactor logic while maintaining the same functionality.
        - Write comments in your own words.
        - Avoid direct copying of large code blocks.

        High similarity increases plagiarism risk; aim for uniqueness.
        """
    elif risk_level == "MEDIUM":
        feedback = f"""
        ### Feedback for '{assignment_name}' (Code File)

        Moderate code similarity detected. Review your code and consider:
        - Adjusting code style and naming.
        - Adding unique functionality or logic improvements.
        - Ensuring proper understanding and re-implementation of logic.
        """
    else:
        feedback = f"""
        ### Feedback for '{assignment_name}' (Code File)

        Low similarity detected. Your code appears original. Keep developing your coding skills and ensuring authenticity.
        """
    return feedback

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def compute_tfidf(docs):
    return TfidfVectorizer(ngram_range=(1, 2), stop_words='english').fit_transform(docs)

def process_assignments():
    all_sub, all_code = {}, {}
    if use_default:
        st.warning("Default folder option selected but folder path input is disabled. Please upload files manually.")
    if uploaded_files:
        for uf in uploaded_files:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext == '.py':
                tmp = f"tmp_{uf.name}"
                with open(tmp, 'wb') as f:
                    f.write(uf.getvalue())
                all_code[uf.name] = extract_code_tokens(tmp)
                os.remove(tmp)
            else:
                txt = extract_text(uploaded_file=uf)
                if txt:
                    all_sub[uf.name] = preprocess_text(txt)
    else:
        st.warning("Please upload files")
        return

    files = list(all_sub.keys())
    code_files = list(all_code.keys())

    all_sents, sent_map = [], []
    for f in files:
        for s in all_sub[f]:
            all_sents.append(s)
            sent_map.append(f)

    sent_sim = None
    if all_sents:
        vs = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
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
        if f in tfidf_scores and f in sem_scores:
            tfidf_scores[f]['grade'] = assign_grade(tfidf_scores[f]['originality_score'])
            sem_scores[f]['grade_sem'] = assign_grade(sem_scores[f]['originality_score'])
            tfidf_scores[f]['combined_grade'] = combined_grade(tfidf_scores[f]['originality_score'], sem_scores[f]['originality_score'])

    code_sim_matrix = None
    code_scores = {}
    if code_files:
        code_sim_matrix = np.zeros((len(code_files), len(code_files)))
        for i, a in enumerate(code_files):
            for j, b in enumerate(code_files):
                if i != j:
                    tokens_a = all_code.get(a, [])
                    tokens_b = all_code.get(b, [])
                    if tokens_a and tokens_b:
                        code_sim_matrix[i, j] = code_similarity(tokens_a, tokens_b)
                    else:
                        code_sim_matrix[i, j] = 0.0
        code_scores = calculate_originality_scores(code_files, code_sim_matrix)
        for f in code_files:
            code_scores[f]['grade'] = code_grade(code_scores[f]['max_similarity'])
            code_scores[f]['combined_grade'] = code_scores[f]['grade']

    st.progress(100)
    st.text("Complete")

    all_files = files + code_files
    summary_rows = []
    for f in all_files:
        if f in code_scores:
            score = code_scores[f]
            summary_rows.append({
                "Assignment": f,
                "Combined Grade": score['combined_grade'],
                "Risk Level": score['risk_level'],
                "Max Similarity": score['max_similarity']
            })
        else:
            score = tfidf_scores[f]
            sem_score = sem_scores[f]
            risk_level = "HIGH" if (score['risk_level'] == "HIGH" or sem_score['risk_level'] == "HIGH") else \
                         "MEDIUM" if (score['risk_level'] == "MEDIUM" or sem_score['risk_level'] == "MEDIUM") else "LOW"
            max_sim = max(score['max_similarity'], sem_score['max_similarity'])
            summary_rows.append({
                "Assignment": f,
                "Combined Grade": score['combined_grade'],
                "Risk Level": risk_level,
                "Max Similarity": max_sim
            })

    summary_df = pd.DataFrame(summary_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Submissions", len(all_files))
    c2.metric("High Risk", (summary_df['Risk Level'] == "HIGH").sum())
    c3.metric("Avg Originality", f"{100 - summary_df['Max Similarity'].mean() * 100:.1f}%")
    c4.metric("High Risk (Semantic)", sum(1 for s in sem_scores.values() if s['risk_level'] == "HIGH"))

    grade_counts = summary_df['Combined Grade'].value_counts().to_dict()
    st.sidebar.markdown("*Grade Distribution (Combined)*")
    for g in ["A", "B", "C", "D", "F"]:
        st.sidebar.write(f"{g}: {grade_counts.get(g, 0)}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary", "📊 Graph", "💻 Code", "📈 Details & Highlights", "📝 Feedback"
    ])

    with tab1:
        st.dataframe(summary_df, use_container_width=True)
        st.download_button("📥 Download CSV", summary_df.to_csv(index=False), "results.csv")

    with tab2:
        st.subheader("Assignment vs Max Similarity")
        fig_bar = go.Figure(go.Bar(
            x=summary_df["Assignment"],
            y=summary_df["Max Similarity"],
            marker_color='indianred'
        ))
        fig_bar.update_layout(
            title="Assignment vs Max Similarity",
            xaxis_title="Assignment",
            yaxis_title="Max Similarity",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        if code_files:
            st.subheader("Token-based Code Similarity Matrix")
            dfc = pd.DataFrame(code_sim_matrix, index=code_files, columns=code_files)
            st.dataframe(dfc.style.background_gradient(cmap='Reds'), use_container_width=True)
        else:
            st.info("No code files for token-based analysis")

    with tab4:
        st.subheader("Details for All Text Files")
        if files:
            for sel in files:
                idx = files.index(sel)
                st.markdown(f"### {sel}")
                st.write("*TF-IDF Top Matches:*")
                tfidf_matches = sorted([(files[j], tfidf_sim[idx][j]) for j in range(len(files)) if j != idx], key=lambda x: -x[1])[:5]
                for fn, sc in tfidf_matches:
                    st.write(f"{fn}: {sc:.4f}")
                st.write("*Semantic Top Matches:*")
                sem_matches = sorted([(files[j], sem_mat[idx][j]) for j in range(len(files)) if j != idx], key=lambda x: -x[1])[:5]
                for fn, sc in sem_matches:
                    st.write(f"{fn}: {sc:.4f}")

                highlights = get_similar_sentences(sel, all_sents, sent_map, sent_sim)
                st.write("*Highlighted Sentences:*")
                disp = [f"<mark>{s}</mark>" if i in highlights else s for i, s in enumerate(all_sents) if sent_map[i] == sel]
                st.markdown(f"<div style='border:1px solid #ddd; padding:10px; max-height:200px; overflow-y:auto;'>{' '.join(disp)}</div>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No text submissions")

    with tab5:
        st.subheader("Plagiarism Improvement Feedback")
        for f in files + code_files:
            if f in files:
                tfidf_risk = tfidf_scores[f]['risk_level']
                sem_risk = sem_scores[f]['risk_level']
                feedback = generate_feedback_text(f, tfidf_risk, sem_risk)
            else:
                max_sim = code_scores[f]['max_similarity'] if f in code_scores else 0.0
                if max_sim >= 0.85:
                    risk_level = "HIGH"
                elif max_sim >= 0.60:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                feedback = generate_feedback_code(f, risk_level)
            st.markdown(feedback)
            st.markdown("---")


if _name_ == "_main_":
    if st.button("🚀 Start Analysis"):
        process_assignments()
    st.markdown("---")
    st.markdown("Assignment Authenticity Monitor")
