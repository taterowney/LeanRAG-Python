from __future__ import annotations
from langchain.globals import set_debug

set_debug(False)

from langchain_core.documents import Document
from langchain_chroma import Chroma

from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
import subprocess
import os, json

# import http.server

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_leanfile_output(cmd=("lake", "exe", "AnnotateImports")):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=ROOT_PATH
    )

    try:
        for line in iter(process.stdout.readline, ''):
            yield line.strip()
        process.stdout.close()
        process.wait()  # Ensure process completes

    except KeyboardInterrupt:
        pass
    finally:
        process.kill()

# Returns tuple of (path, [annotated_theorems])
def annotated_thms_generator_stdio():
    for line in get_leanfile_output():
        try:
            data = json.loads(line)
            yield data["filename"], data["theorems"]
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {line}")
            continue


def make_import_database(number_to_retrieve=6, filter=None):
    embeddings = OllamaEmbeddings(model="llama3.2")
    gen = annotated_thms_generator_stdio()
    # gen = annotated_thms_generator_http()
    vectorstore = Chroma(collection_name="Annotated_Mathlib_Theorems", embedding_function=embeddings, persist_directory=DB_PATH)

    docs_buffer = []
    for (file_name, annotated_theorems) in gen:
        for thm in annotated_theorems:
            doc = Document(
                page_content=thm,
                metadata={"file": file_name},
            )
            docs_buffer.append(doc)
    vectorstore.add_documents(docs_buffer)
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": number_to_retrieve, "filter": filter})