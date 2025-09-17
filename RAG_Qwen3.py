
# 0. Importar Librerías
from litellm import completion
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import pdfplumber
import uuid
from typing import List, Tuple

# 1. Cargar model de embedding
model_name = "Qwen/Qwen3-Embedding-0.6B"
embedder = SentenceTransformer(model_name, device="cuda")

def embed_text(text: str):
    """
    Genera embeddings con Qwen3-Embedding-0.6B en GPU
    """
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()

# 2. Inicializar ChromaDB
# En memoria
client = chromadb.Client()
collection = client.get_or_create_collection(name="docs")
# Persistente
# client = chromadb.PersistentClient(path="./chroma_db")  
# collection = client.create_collection("mi_coleccion")

# 3. Ingestar documentos

def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """Devuelve lista de tuplas (número_de_página, texto_de_página)."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            pages.append((i + 1, txt))
    return pages

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunking simple por palabras con solapamiento (caracteres aproximados)."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for w in words:
        current.append(w)
        current_len += len(w) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # preparar solapamiento
            if overlap > 0:
                # tomar las últimas palabras que aproximan el overlap (por caracteres)
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

def ingest_pdf_to_chromadb(pdf_path: str, collection, chunk_size: int = 1000, overlap: int = 200):
    """
    Extrae texto del PDF, lo chunkea, calcula embeddings (usa embed_text) y los añade a collection.
    collection: objeto chromadb collection ya instanciado.
    """
    pages = extract_text_from_pdf(pdf_path)
    total = 0
    for page_number, page_text in pages:
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            emb = embed_text(chunk)  # usa la función ya definida en el notebook
            doc_id = str(uuid.uuid4())
            # añadir con metadatos útiles
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[doc_id],
                metadatas=[{"source": pdf_path, "page": page_number}]
            )
            total += 1
    print(f"Ingestados {total} chunks desde {pdf_path} en la colección '{collection.name}'.")

ingest_pdf_to_chromadb("C:/Users/josue/Downloads/Propuesta Políticas de Gobierno de Datos.pdf", collection, chunk_size=1000, overlap=200)

# 4. Recuperar contexto
def retrieve_context(query: str, n_results: int = 3) -> list:
    query_emb = embed_text(query)
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)
    return " ".join(results['documents'][0])

# 5. RAG con Ollama
def rag_query(query: str) -> str:
    context = retrieve_context(query)
    promtp = f"""
    Usa el siguiente contexto para responder la pregunta:

    Contexto: {context}

    Pregunta: {query}
    """
    response = completion(
        model="ollama/Qwen3:8B",
        messages=[{"role": "user", "content": promtp}],
        api_base="http://localhost:11434",
        temperature=0.1,
        top_p=0.7,
        top_k=40,
        stream=True # Habilitar el streaming
    )
    # `response` ahora es un generador
    for chunk in response:
        # Cada chunk contiene parte del texto
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
    print()  # salto de línea al final

# 6. Preguntar
if __name__ == "__main__":
    pregunta = input("Escribe tu pregunta: ")
    #print("Pregunta:", pregunta)
    #respuesta = rag_query(pregunta)
    print("Respuesta:", end=" ")
    rag_query(pregunta)
