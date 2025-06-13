import os, json, subprocess, re
import time

from .LeanIO import check_leanRAG_installation
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .utils import load_annotated_goal_state_theorems, load_plain_theorems
from .annotate import get_goal_annotations

class __version__:
    def __init__(self, version):
        self.version = version

    def __str__(self):
        return self.version
__version__ = __version__("0.1.0")


class Retriever:
    def __init__(self, database_path=None, model=None, preprocess=load_plain_theorems(["Mathlib"])):

        if not database_path or not os.path.exists(database_path):
            if not model or not preprocess:
                raise ValueError("Must specify a retrieval model and a preprocessor to create a new vectorstore")

            if not database_path:
                self.database_path = os.path.join(
                    ".db",
                    f"{time.time()}_{model.lower()}"
                )
            else:
                self.database_path = database_path
            os.makedirs(self.database_path, exist_ok=True)

            self.model = model
            self.preprocess = preprocess
            self.created = False
            self._set_config()

        else:
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
        for declaration in self.preprocess():
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

# if __name__ == "__main__":
#     check_leanRAG_installation(project_dir="../../test_project")
#     print(get_goal_annotations("Mathlib.Algebra.AddConstMap.Basic", project_dir="../../test_project"))
