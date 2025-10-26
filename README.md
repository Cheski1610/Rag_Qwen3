# **RAG_Qwen3**

Este repositorio contiene ejemplos y utilidades para construir un sistema RAG (Retrieval-Augmented Generation) usando el modelo Qwen3 para embeddings y distintos backends de generación (Ollama local, GitHub Models o Cerebras). A continuación se describen los archivos principales, su funcionalidad y las diferencias entre ellos.

## Archivos

- `RAG_Qwen3.py`
  - Propósito: Script de Python autónomo pensado para ejecución desde línea de comandos.
  - Funcionalidad principal:
    - Carga un modelo de embeddings (`SentenceTransformer`) y lo usa para generar embeddings de fragmentos de texto.
    - Inicializa una colección de ChromaDB en memoria y proporciona funciones para extraer texto de PDFs, fragmentarlo (chunking) y añadir los chunks con embeddings a la colección.
    - Implementa una función de recuperación de contexto simple (`retrieve_context`) y un flujo RAG (`rag_query`) que envía una prompt al modelo de generación (`completion`) y muestra la respuesta por consola en streaming.
  - Uso esperado: ejecutar desde consola para ingestar un PDF local y probar consultas interactivas.
  - Notas:
    - Puedes ajustar temperatura, `top_k`, `top_p` y el proveedor desde variables de entorno (`GEN_TEMPERATURE`, `GEN_TOP_K`, `GEN_TOP_P`, `GEN_MODEL_PROVIDER`).
    - Soporta los mismos backends de generación que la app (Ollama, GitHub Models y Cerebras).
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
    - Permite seleccionar el backend de generación (Ollama, GitHub Models o Cerebras) y configurar modelo/endpoint/token desde la UI.
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
  - En `RAG_Qwen3.py` puedes ajustar proveedor y sampling mediante variables de entorno (`GEN_*`).
  - En `app_streamlit.py` están expuestos en la UI y `top_k=0` sigue omitiéndose del payload cuando no se usa.

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

## Uso en GitHub Codespaces

El repositorio incluye una carpeta `.devcontainer/` lista para Codespaces:

- Al abrir el repo en GitHub Codespaces, VS Code utilizará la imagen definida en `Dockerfile` (Python 3.12) y ejecutará `pip install -r requirements.txt` automáticamente.
- El puerto 8501 ya está reenviado; ejecuta `streamlit run app_streamlit.py` para exponer la UI y usa el panel **Ports** para abrirla en el navegador.
- Codespaces no trae Ollama instalado; selecciona desde la app un backend remoto (GitHub Models o Cerebras) o ajusta `GEN_MODEL_PROVIDER` en el CLI.
- Configura credenciales sensibles como **Codespaces secrets** antes de abrir el entorno (`GITHUB_MODELS_TOKEN`, `CEREBRAS_API_KEY`, etc.); el contenedor las leerá como variables de entorno.
- Si necesitas comandos adicionales tras la creación, añade un script a `postCreateCommand` o ejecuta tareas manualmente dentro del Codespace.

## Proveedores de modelos y credenciales

La aplicación de Streamlit y el script CLI pueden trabajar con varios backends de generación:

- **Ollama (local)**: configuración por defecto. Personaliza modelo y endpoint con `OLLAMA_MODEL_ID` y `OLLAMA_API_BASE`.
- **GitHub Models**: define un token con `GITHUB_MODELS_TOKEN` (o `st.secrets`) y, si lo necesitas, ajusta `GITHUB_MODEL_ID` y `GITHUB_MODELS_API_BASE`.
- **Cerebras**: establece `CEREBRAS_API_KEY` (o `st.secrets`) y opcionalmente `CEREBRAS_MODEL_ID` / `CEREBRAS_API_BASE`.

Para reutilizar la misma configuración en el script, fija `GEN_MODEL_PROVIDER` en `ollama`, `github` o `cerebras`. Parámetros adicionales:

- `GEN_TEMPERATURE`, `GEN_TOP_P` y `GEN_TOP_K` (solo aplica a Ollama) para ajustar la decodificación.
- `GEN_CONTEXT_RESULTS` para definir cuántos chunks recuperar desde ChromaDB.

---