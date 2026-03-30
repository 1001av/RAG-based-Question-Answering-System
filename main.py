import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def create_openai_client():
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Warning: no GROQ_API_KEY or OPENAI_API_KEY found. "
            "Answer generation will return retrieved context only."
        )
        return None

    if os.getenv("GROQ_API_KEY"):
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    return OpenAI(api_key=api_key)


client = create_openai_client()

# -------- LOAD DOCUMENTS --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")

documents = []

for filename in os.listdir(DATA_FOLDER):
    path = os.path.join(DATA_FOLDER, filename)
    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif filename.lower().endswith(".txt"):
        loader = TextLoader(path, encoding="utf8")
    else:
        continue

    try:
        documents.extend(loader.load())
    except Exception as exc:
        print(f"Warning: could not load {filename}: {exc}")

if not documents:
    raise SystemExit(
        "No documents loaded. Add .pdf or .txt files to the data/ folder and rerun."
    )

print(f"Loaded {len(documents)} pages from data files")

# -------- CHUNKING --------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]

if not texts:
    raise SystemExit("No text chunks were created. Check your data files and document loader.")

print(f"Created {len(texts)} chunks")

# -------- EMBEDDINGS --------
def get_embedding(text):
    return np.random.rand(384)

embeddings = np.array([get_embedding(t) for t in texts], dtype="float32")

# -------- FAISS INDEX --------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------- RETRIEVAL --------
def retrieve(query, k=3):
    if not texts:
        return []

    query_vector = np.array([get_embedding(query)], dtype="float32")
    k = min(k, len(texts))
    distances, indices = index.search(query_vector, k)
    return [texts[i] for i in indices[0] if i >= 0]


# -------- GENERATE ANSWER --------
def ask(query):
    context = "\n\n".join(retrieve(query))
    if not client:
        return (
            "No API key configured. Showing retrieved context only.\n\n" + context
        )

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question: {query}
Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    choice = response.choices[0]
    if hasattr(choice, "message"):
        message = choice.message
        return getattr(message, "content", None) or message["content"]
    return getattr(choice, "text", str(choice))


# -------- RUN --------
if __name__ == "__main__":
    print("Type a question and press Enter. Type 'exit' or 'quit' to stop.")
    while True:
        q = input("\nAsk: ").strip()
        if q.lower() in {"", "exit", "quit"}:
            print("Goodbye.")
            break
        print("\nAnswer:\n", ask(q))