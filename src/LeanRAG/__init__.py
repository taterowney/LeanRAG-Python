import json
import os
import re
import subprocess
import time
from pathlib import Path

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
    """Simple wrapper around a Chroma vector store."""

    def __init__(self, database_path: str | Path | None = None, model=None, preprocess=load_plain_theorems(["Mathlib"])):
        """Initialize a new :class:`Retriever`.

        Parameters
        ----------
        database_path : str | Path | None, optional
            Directory where the vector store should be stored.  If ``None`` a
            new directory is created.
        model : str | None
            Name of the embedding model used by the vector store.
        preprocess : Callable
            Callable returning an iterable of documents to index when creating
            a new store.
        """

        if not database_path or not Path(database_path).exists():
            if not model or not preprocess:
                raise ValueError("Must specify a retrieval model and a preprocessor to create a new vectorstore")

            if not database_path:
                self.database_path = Path(".db") / f"{time.time()}_{model.lower().replace('/', '_')}"
            else:
                self.database_path = Path(database_path)
            self.database_path.mkdir(parents=True, exist_ok=True)

            self.model = model
            self.preprocess = preprocess
            self.created = False
            self._set_config()

        else:
            self.database_path = Path(database_path)
            if any(p.suffix == ".sqlite3" for p in self.database_path.iterdir()):
                self.created = True
            else:
                self.created = False

            self.preprocess = preprocess
            if model:
                self.model = model
                self._set_config()
            else:
                self.model = self._get_config()["model"]


        self.vectorstore = None

    def _get_config(self):
        """Return the configuration stored alongside the vector store."""
        config_file = self.database_path / "config.json"
        if not config_file.exists():
            raise ValueError("No config.json file found in the database directory. Specify a model and a preprocessor to create or convert a vectorstore.")

        with config_file.open("r") as f:
            config = json.load(f)

        return config

    def _set_config(self):
        """Persist configuration information for the vector store."""
        config = {
            "model": self.model,
            "preprocess": self.preprocess.__name__
        }
        with (self.database_path / "config.json").open("w") as f:
            json.dump(config, f, indent=4)

    def create_vectorstore(self):
        """Create a new Chroma vector store and ingest documents."""

        embeddings = HuggingFaceEmbeddings(
            model_name=self.model
        )
        vectorstore = Chroma(
            collection_name=self.model.lower().replace("/", "_") + "_db",
            persist_directory=str(self.database_path),
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
        """Load an existing Chroma vector store from disk."""
        if not self.created:
            raise ValueError("Vectorstore not created. Call create_vectorstore() first.")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.model
        )
        self.vectorstore = Chroma(
            collection_name=self.model.lower().replace("/", "_") + "_db",
            persist_directory=str(self.database_path),
            embedding_function=embeddings,
        )

    def retrieve(self, query, k=1, search_type="mmr"):
        """Retrieve documents relevant to *query* from the vector store."""
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
