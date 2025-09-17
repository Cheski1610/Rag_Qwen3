# RAG_Qwen3

Este repositorio contiene ejemplos y utilidades para construir un sistema RAG (Retrieval-Augmented Generation) usando el modelo Qwen3 para embeddings y Ollama/Qwen3 para generación. A continuación se describen los archivos principales, su funcionalidad y las diferencias entre ellos.

## Archivos

- `RAG_Qwen3.py`
  - Propósito: Script de Python autónomo pensado para ejecución desde línea de comandos.
  - Funcionalidad principal:
    - Carga un modelo de embeddings (`SentenceTransformer`) y lo usa para generar embeddings de fragmentos de texto.
    - Inicializa una colección de ChromaDB en memoria y proporciona funciones para extraer texto de PDFs, fragmentarlo (chunking) y añadir los chunks con embeddings a la colección.
    - Implementa una función de recuperación de contexto simple (`retrieve_context`) y un flujo RAG (`rag_query`) que envía una prompt al modelo de generación (`completion`) y muestra la respuesta por consola en streaming.
  - Uso esperado: ejecutar desde consola para ingestar un PDF local y probar consultas interactivas.
  - Notas:
    - En el script se usan valores explícitos como `top_k=40` en la llamada a `completion`.
    - Está orientado a un flujo de demostración y a pasos manuales (no está integrado con UI).

- `RAG_Qwen3.ipynb`
  - Propósito: Notebook interactivo para exploración, experimentación y demostración.
  - Funcionalidad principal:
    - Contiene celdas que cubren los mismos pasos conceptuales que el script (`RAG_Qwen3.py`): carga de modelos, inicialización de ChromaDB, funciones de extracción/chunking/ingesta, recuperación de contexto y ejemplo de consulta RAG.
    - Diseñado para ser ejecutado por partes en un entorno Jupyter — útil para depuración, pruebas de snippets y visualización de resultados intermedios.
  - Uso esperado: abrir con Jupyter/VS Code y ejecutar celdas de forma interactiva.
  - Notas:
    - Ideal para experimentar con parámetros (chunk_size, overlap, n_results) y para inspeccionar memoria/colecciones en tiempo real.

- `app_streamlit.py`
  - Propósito: Interfaz web interactiva construida con Streamlit para ingestar PDFs y realizar consultas RAG desde una UI amigable.
  - Funcionalidad principal:
    - Permite subir un PDF desde el navegador y guarda un temporal local.
    - Carga (con cache) un modelo de embeddings y crea una colección de ChromaDB donde ingesta los chunks del PDF con embeddings.
    - Panel lateral con controles ajustables: temperatura, `top_k`, `top_p`, seed y número de resultados a recuperar.
    - Construye consultas RAG que combinan el contexto recuperado (desde ChromaDB) con una prompt enviada al modelo de generación y muestra la respuesta en streaming en la UI.
  - Diferencias importantes respecto a los otros archivos:
    - `app_streamlit.py` es una aplicación web con UI y estado de sesión; los otros son scripts/notebooks sin interfaz web.
    - Implementa manejo de estado (`st.session_state`) para mantener la colección, el cliente y el embedder entre interacciones.
    - Ajustes de parámetros en tiempo real (ej. temperatura, top_k/top_p) y estilo visual.
    - Reciente cambio: `top_k` ahora tiene valor por defecto `0` en la UI y se omite del diccionario de parámetros enviados al generador cuando su valor es `0` — esto evita pasar `top_k` no deseado al API cuando el usuario no lo habilita.

## Diferencias resumidas
- Interfaz:
  - `app_streamlit.py` → Web UI (Streamlit)
  - `RAG_Qwen3.py` → Script CLI
  - `RAG_Qwen3.ipynb` → Notebook interactivo

- Propósito de uso:
  - `app_streamlit.py` → demostración interactiva y uso por usuarios finales o pruebas rápidas con UI.
  - `RAG_Qwen3.py` → pruebas en línea de comandos, automatización o integración sencilla en pipelines.
  - `RAG_Qwen3.ipynb` → exploración, experimentación y documentación reproducible.

- Manejo de parámetros:
  - En `RAG_Qwen3.py` muchos parámetros están codificados (por ejemplo `top_k=40` en la llamada `completion`).
  - En `app_streamlit.py` están expuestos en la UI y ahora `top_k=0` es el valor por defecto (omisión si 0).

## Recomendaciones rápidas
- Si quieres una demo rápida con UI, ejecuta:
```powershell
streamlit run .\app_streamlit.py
```

- Para ejecutar el script de ejemplo desde consola:
```powershell
python .\RAG_Qwen3.py
```

- Para experimentar y documentar pasos reproductibles, abre `RAG_Qwen3.ipynb` en Jupyter o en VS Code.

---