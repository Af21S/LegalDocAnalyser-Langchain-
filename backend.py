from typing import List, Dict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline


# -------------------- PDF LOADING --------------------

def load_pdf(path: str) -> List[Document]:
    return PyPDFLoader(path).load()


def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# -------------------- EMBEDDINGS & VECTOR STORE --------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vectorstore(chunks: List[Document]) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


# -------------------- LLM SETUP (CPU-FRIENDLY) --------------------

model_id = "Qwen/Qwen2.5-3B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",       # Force CPU (fast & avoids GPU issues)
    torch_dtype="auto",
    trust_remote_code=True
)

# Pipeline with reduced tokens for speed
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,      # Reduced from 500 -> fast CPU test
    temperature=0,
    do_sample=False
)

# HuggingFacePipeline wrapper
llm = HuggingFacePipeline(pipeline=pipe)


# -------------------- PROMPT --------------------

prompt = PromptTemplate.from_template("""
You are a STRICT Indian legal contract analysis system.

RULES (NO EXCEPTIONS):
- Use ONLY the provided context
- Quote clauses EXACTLY
- Mention section / clause numbers if present
- If missing, respond ONLY: "Not specified"
- Do NOT explain concepts
- Do NOT invent examples
- Do NOT add external law

OUTPUT FORMAT:

CONFIDENTIALITY:
"Exact quoted clause" OR Not specified

TERMINATION:
"Exact quoted clause" OR Not specified

LIABILITY:
"Exact quoted clause" OR Not specified

JURISDICTION / GOVERNING LAW:
"Exact quoted clause" OR Not specified

INTELLECTUAL PROPERTY:
"Exact quoted clause" OR Not specified

PENALTIES / DAMAGES:
"Exact quoted clause" OR Not specified

Context:
{context}

Question:
{question}
""")


# -------------------- CLAUSE QUERIES --------------------

CLAUSE_QUERIES = {
    "Confidentiality": "confidentiality non disclosure secrecy",
    "Termination": "termination cancellation expiry",
    "Liability": "liability indemnity damages",
    "Jurisdiction": "jurisdiction governing law courts",
    "Intellectual Property": "intellectual property copyright patent ownership",
    "Penalties": "penalty damages fine compensation"
}


# -------------------- CORE LOGIC --------------------

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def analyze_clauses(vectorstore: FAISS) -> Dict[str, str]:
    results = {}

    for clause, query in CLAUSE_QUERIES.items():
        docs = vectorstore.similarity_search(query, k=4)
        context = format_docs(docs)

        # Take only first 2000 chars of context for CPU speed
        context_chunk = context[:2000]

        response = llm.invoke(
            prompt.format(
                context=context_chunk,
                question=f"Extract {clause} clause."
            )
        )

        results[clause] = response

    return results


# -------------------- HALLUCINATION CHECK --------------------

def context_overlap(context: str, generated: str) -> float:
    context_tokens = set(context.split())
    generated_tokens = generated.split()

    if not generated_tokens:
        return 0.0

    return sum(1 for t in generated_tokens if t in context_tokens) / len(generated_tokens)


# -------------------- MAIN API FOR STREAMLIT --------------------

def analyze_contract(pdf_path: str):
    """
    Main entry point used by Streamlit UI.

    Returns:
        results (Dict[str, str]): clause-wise extracted text
        overlaps (Dict[str, float]): hallucination scores
    """

    docs = load_pdf(pdf_path)
    chunks = chunk_docs(docs)
    vectorstore = build_vectorstore(chunks)

    results = analyze_clauses(vectorstore)

    full_context = " ".join(chunk.page_content for chunk in chunks)

    overlaps = {
        clause: context_overlap(full_context, text)
        for clause, text in results.items()
    }

    return results, overlaps
