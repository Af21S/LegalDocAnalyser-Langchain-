import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from backend import analyze_contract



# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# -------------------- HEADER --------------------
st.markdown("""
# ⚖️ Legal Document Analyzer  
**Indian Contract Clause Extractor & Hallucination Checker**

Upload a legal PDF to extract *exact contractual clauses*.  
_No explanations. No hallucinations._
""")

st.divider()


# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "📄 Upload a legal contract (PDF only)",
    type=["pdf"]
)

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Run analysis
    with st.spinner("Analyzing contract… please wait ⏳"):
        results, overlaps = analyze_contract(pdf_path)

    st.success("Analysis complete ✅")

    # -------------------- RESULTS --------------------
    st.subheader("📌 Clause-wise Extraction")

    for clause, text in results.items():
        with st.expander(f"🔹 {clause.upper()}"):
            st.code(text, language="text")

    # -------------------- HALLUCINATION CHECK --------------------
    st.subheader("🧪 Hallucination Check")

    fig, ax = plt.subplots()
    ax.bar(overlaps.keys(), overlaps.values())
    ax.set_ylabel("Context Overlap Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30)

    st.pyplot(fig)

    # -------------------- RAW SCORES --------------------
    with st.expander("📊 Raw Overlap Scores"):
        st.json(overlaps)
