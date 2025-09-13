import streamlit as st
from litellm import completion
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import pdfplumber
import uuid
import tempfile
from typing import List, Tuple


st.set_page_config(page_title="RAG Qwen3 - Streamlit", layout="wide")

# Model names (mostrados de forma estática en la UI)
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL_NAME = "ollama/Qwen3:8B"


@st.cache_resource
def load_embedder(model_name: str, device: str):
    """
    Intenta cargar el modelo de embeddings por id (Hugging Face). Si falla, intenta cargar desde carpetas locales conocidas
    (p. ej. 'Qwen3-0.6B') y devuelve una tupla (embedder, ruta_usada).
    """
    import os
    tried = []
    # intento 1: modelo remoto/identificador
    try:
        emb = SentenceTransformer(model_name, device=device)
        return emb, model_name
    except Exception as e:
        tried.append(f"{model_name}: {e}")

    # intento 2: buscar carpeta local en workspace
    local_candidates = [
        os.path.join(os.getcwd(), "Qwen3-0.6B"),
        os.path.join(os.getcwd(), "Qwen3", "Qwen3-Embedding-0.6B"),
        "Qwen3-0.6B",
    ]
    for cand in local_candidates:
        if os.path.exists(cand):
            try:
                emb = SentenceTransformer(cand, device=device)
                return emb, cand
            except Exception as e:
                tried.append(f"{cand}: {e}")

    # si llegamos aquí, no se pudo cargar nada
    raise RuntimeError("No se pudo cargar el modelo de embeddings. Intentos: " + " | ".join(tried))


def embed_text(embedder, text: str):
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages.append((i + 1, txt))
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for w in words:
        current.append(w)
        current_len += len(w) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            if overlap > 0:
                ov_words = []
                ov_len = 0
                for tok in reversed(current):
                    ov_words.insert(0, tok)
                    ov_len += len(tok) + 1
                    if ov_len >= overlap:
                        break
                current = ov_words.copy()
                current_len = sum(len(tok) + 1 for tok in current)
            else:
                current = []
                current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest_pdf_to_chromadb(pdf_path: str, collection, embedder, chunk_size: int = 1000, overlap: int = 200):
    pages = extract_text_from_pdf(pdf_path)
    total = 0
    for page_number, page_text in pages:
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            emb = embed_text(embedder, chunk)
            doc_id = str(uuid.uuid4())
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[doc_id],
                metadatas=[{"source": pdf_path, "page": page_number}]
            )
            total += 1
    return total


def retrieve_context(collection, embedder, query: str, n_results: int = 3) -> str:
    query_emb = embed_text(embedder, query)
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)
    docs = results.get('documents', [])
    if docs and len(docs) > 0:
        # documents usually a list of lists
        first = docs[0]
        if isinstance(first, list):
            return " ".join(first)
        return str(first)
    return ""


def rag_query_stream(question: str, collection, embedder, temperature: float, top_k: int = None, top_p: float = None, seed: int = None):
    context = retrieve_context(collection, embedder, question)
    prompt = f"""
    Usa el siguiente contexto para responder la pregunta:

    Contexto: {context}

    Pregunta: {question}
    """
    # Construir kwargs dinámicamente para no pasar parámetros no deseados
    params = {
        "model": GEN_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "api_base": "http://localhost:11434",
        "temperature": temperature,
        "stream": True,
    }
    # sólo incluir top_k si se ha especificado y es mayor que 0
    if top_k is not None and int(top_k) > 0:
        params["top_k"] = int(top_k)
    if top_p is not None:
        params["top_p"] = top_p
    if seed is not None and seed >= 0:
        params["seed"] = int(seed)

    response = completion(**params)
    # generator -> yield incremental text
    for chunk in response:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            yield delta


def main():
    # Estilos personalizados con color base #2991AB y degradados
    st.markdown(
        '''
        <style>
        body, .main, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #eaf6fa 0%, #2991AB 150%);
        }
        [data-testid="stSidebar"], .stSidebar {
            background: #eaf6fa !important;
            color: #17647A;
        }
        .stButton > button {
            background: linear-gradient(90deg, #2991AB 0%, #17647A 100%);
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 0.5em 1.2em;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(41,145,171,0.15);
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #17647A 0%, #2991AB 100%);
            color: #fff;
        }
        .stTextArea textarea {
            background: #eaf6fa;
            border: 1px solid #2991AB;
        }
        .stHeader, .stTitle, h1, h2, h3, h4 {
            color: #17647A;
        }
        .stSuccess, .stAlert-success {
            background: #2991AB22;
            color: #17647A;
            border-left: 5px solid #2991AB;
        }
        .stWarning, .stAlert-warning {
            background: #ffe6e6;
            color: #17647A;
            border-left: 5px solid #2991AB;
        }
        .stInfo, .stAlert-info {
            background: #eaf6fa;
            color: #17647A;
            border-left: 5px solid #2991AB;
        }
        .stMarkdown, .markdown-text-container {
            color: #17647A;
        }
        </style>
        ''', unsafe_allow_html=True
    )
    st.title("RAG con Qwen3 — Streamlit")

    # Sidebar controls
    st.sidebar.header("Configuración")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"Dispositivo detectado: {device}")
    st.sidebar.markdown(f"**Modelo:** {GEN_MODEL_NAME}")
    temp = st.sidebar.slider("Temperatura", min_value=0.0, max_value=2.0, value=0.1, step=0.1)
    top_k = st.sidebar.slider("top_k (núm. de tokens candidatos)", min_value=0, max_value=100, value=0, step=1)
    top_p = st.sidebar.slider("top_p (probabilidad de tokens acumulada)", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
    seed_input = st.sidebar.number_input("Seed (enter -1 para aleatorio/no establecido)", min_value=-1, value=-1, step=1)
    n_results = st.sidebar.number_input("Resultados a recuperar", min_value=1, max_value=10, value=3, step=1)

    uploaded = st.file_uploader("Sube un PDF para ingestar", type=["pdf"])

    if 'client' not in st.session_state:
        st.session_state.client = None
        st.session_state.collection = None
        st.session_state.embedder = None
        st.session_state.embedder_source = None
        st.session_state.ready = False
        st.session_state.collection_name = None

    if uploaded is not None:
        with st.spinner("Guardando archivo temporal..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            tmp_path = tmp.name
        st.success(f"Archivo guardado en {tmp_path}")

        if st.button("Ingestar PDF en ChromaDB"):
            with st.spinner("Cargando modelo de embeddings..."):
                embedder, embedder_source = load_embedder(EMBED_MODEL_NAME, device)
                st.session_state.embedder = embedder
                st.session_state.embedder_source = embedder_source
            client = chromadb.Client()
            collection_name = f"docs_{uuid.uuid4().hex}"
            collection = client.get_or_create_collection(name=collection_name)
            st.session_state.client = client
            st.session_state.collection = collection
            st.session_state.collection_name = collection_name
            st.session_state.ready = True
            with st.spinner("Extrayendo e ingestado... Esto puede tardar según el PDF y el modelo"):
                cnt = ingest_pdf_to_chromadb(tmp_path, collection, embedder, chunk_size=1000, overlap=200)
            st.success(f"Ingestados {cnt} chunks en la colección '{collection_name}'")

    st.markdown("---")
    st.header("Preguntar al modelo")
    question = st.text_area("Escribe tu pregunta aquí", height=120)

    # Mostrar la fuente del embedder si está disponible
    if st.session_state.get('embedder_source'):
        st.sidebar.info(f"Embedder cargado desde: {st.session_state.embedder_source}")

    if st.button("Enviar pregunta"):
        if not question.strip():
            st.warning("Escribe una pregunta antes de enviar.")
        elif not st.session_state.get('ready'):
            st.warning("No hay ninguna colección ingested. Sube e ingesta un PDF primero.")
        else:
            placeholder = st.empty()
            output = ""
            with st.spinner("Solicitando respuesta al modelo..."):
                for delta in rag_query_stream(question, st.session_state.collection, st.session_state.embedder, temp, top_k=top_k, top_p=top_p, seed=seed_input if seed_input >= 0 else None):
                    output += delta
                    placeholder.markdown(output)
            st.success("Respuesta completa")

    # Mostrar estado de la colección en la sidebar
    if st.session_state.get('ready'):
        st.sidebar.success(f"Colección lista: {st.session_state.get('collection_name')}")
    else:
        st.sidebar.info("No hay colección ingested aún.")


if __name__ == "__main__":
    main()
