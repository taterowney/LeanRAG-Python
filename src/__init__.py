# from retrieve import get_database_retriever
import os, json, subprocess, re
import time

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class __version__:
    def __init__(self, version):
        self.version = version

    def __str__(self):
        return self.version
__version__ = __version__("0.1.0")

def get_decls_from_plaintext(text):
    """
    Extract declarations from a plaintext Lean file.
    This function assumes that declarations are separated by newlines.
    """

    lines = text.splitlines(keepends=True)

    # Match lines like:
    #   theorem …
    #   @[simp] lemma …
    #   @[attr1][@attr2] def …
    decl_re = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)*(theorem|lemma|example|problem|def)\b')

    # find every line index where a target decl begins
    decl_starts = [i for i, ln in enumerate(lines) if decl_re.match(ln)]

    blocks = []
    for idx, start in enumerate(decl_starts):
        end = decl_starts[idx + 1] if idx + 1 < len(decl_starts) else len(lines)

        # walk backwards to include any preceding comments or attributes
        j = start - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        while j >= 0 and (lines[j].lstrip().startswith('--')
                          or lines[j].lstrip().startswith('@['))\
                          or lines[j].lstrip().startswith('/--'):
            j -= 1

        block_lines = lines[j + 1:end]

        # compute the indentation of the declaration line
        # indent = re.match(r'^(\s*)', lines[start]).group(1)
        # decl_indent = len(indent.expandtabs())

        # now trim off any trailing blank lines or `end …` at ≤ that indent
        if not block_lines:
            continue

        has_started = False
        for i, line in enumerate(block_lines):
            if decl_re.match(line):
                has_started = True
                continue
            if has_started and not line.startswith("  "):
                break
        block_lines = block_lines[:i]

        if block_lines:
            if "".join(block_lines).strip():
                blocks.append("".join(block_lines).strip())

    return blocks


def load_lean(modules):

    paths = []
    cmd = "find .lake/packages/ -type f -name '*.lean'"
    lake_packages = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout.split("\n")
    lake_packages = {
        "/".join(p.split(".lake/packages/")[1].split("/")[1:]) : os.path.join(os.getcwd(), p) for p in lake_packages
    }
    for m in modules:
        as_path = m.replace(".", "/")
        if os.path.exists(os.path.join(os.getcwd(), as_path + ".lean")):
            paths.append(os.path.join(os.getcwd(), as_path + ".lean"))
        elif as_path + ".lean" in lake_packages.keys():
            paths.append(lake_packages[as_path + ".lean"])

        # Handle directories
        if os.path.exists(os.path.join(os.getcwd(), as_path)):
            for root, _, files in os.walk(os.path.join(os.getcwd(), as_path)):
                for file in files:
                    if file.endswith(".lean"):
                        paths.append(os.path.join(root, file))
        elif as_path in lake_packages.keys():
            for root, _, files in os.walk(lake_packages[as_path]):
                for file in files:
                    if file.endswith(".lean"):
                        paths.append(os.path.join(root, file))

    paths = list(set(paths))

    for path in paths:
        assert os.path.exists(path), f"File {path} does not exist. Tate made an oopsie"
        with open(path, "r") as f:
            for decl in get_decls_from_plaintext(f.read()):
                yield decl

class Retriever:
    def __init__(self, modules=(), database_path=None, model=None, preprocess=load_lean):

        if not database_path:
            if not model or not preprocess:
                raise ValueError("Must specify a retrieval model and a preprocessor to create a new vectorstore")

            self.database_path = os.path.join(
                ".db",
                f"{time.time()}_{model.lower()}"
            )
            os.makedirs(self.database_path, exist_ok=True)

            self.model = model
            self.preprocess = preprocess
            self.created = False
            self._set_config()

        else:
            if not os.path.exists(database_path):
                raise ValueError(f"No database found at {database_path}")
            self.database_path = database_path
            if any([file.endswith(".sqlite3") for file in os.listdir(self.database_path)]):
                self.created = True
            else:
                self.created = False

            if model:
                self._set_config()
            else:
                self.model = self._get_config()["model"]

            self.preprocess = preprocess

        self.modules = modules
        self.vectorstore = None

    def _get_config(self):
        if not os.path.exists(os.path.join(self.database_path, "config.json")):
            raise ValueError("No config.json file found in the database directory. Specify a model and a preprocessor to create or convert a vectorstore.")

        with open(os.path.join(self.database_path, "config.json"), "r") as f:
            config = json.load(f)

        return config

    def _set_config(self):
        config = {
            "model": self.model,
            "preprocess": self.preprocess.__name__
        }
        with open(os.path.join(self.database_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    def create_vectorstore(self):

        embeddings = HuggingFaceEmbeddings(
            model_name=self.model
        )
        vectorstore = Chroma(
            collection_name=self.model.lower() + "_db",
            persist_directory=self.database_path,
            embedding_function=embeddings,
        )

        docs_queue = []
        for declaration in self.preprocess(self.modules):
            if type(declaration) is str:
                docs_queue.append(
                    Document(
                        page_content=declaration,
                    )
                )
            elif type(declaration) is Document:
                docs_queue.append(declaration)
            elif type(declaration) is dict:
                docs_queue.append(
                    Document(
                        page_content=declaration["page_content"],
                        metadata=declaration.get("metadata", {})
                    )
                )

            if len(docs_queue) >= 1000:
                vectorstore.add_documents(docs_queue)
                docs_queue = []

        self.created = True

        self.vectorstore = vectorstore

    def load_vectorstore(self):
        if not self.created:
            raise ValueError("Vectorstore not created. Call create_vectorstore() first.")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.model
        )
        self.vectorstore = Chroma(
            collection_name=self.model.lower() + "_db",
            persist_directory=self.database_path,
            embedding_function=embeddings,
        )

    def retrieve(self, query, k=1, search_type="mmr"):
        if not self.vectorstore:
            if not self.created:
                print("Vectorstore not created. Creating a new vectorstore...")
                self.create_vectorstore()
            else:
                self.load_vectorstore()

        assert self.vectorstore is not None, "Vectorstore is not loaded or created."

        retriever = self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )

        results = retriever.invoke(query)
        return results