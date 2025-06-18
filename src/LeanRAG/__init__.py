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

from .utils import load_annotated_goal_state_theorems, load_plain_theorems, get_all_modules
from .annotate import get_goal_annotations
from typing import Tuple

class __version__:
    def __init__(self, version):
        self.version = version

    def __str__(self):
        return self.version
__version__ = __version__("0.1.0")

def _list_configs(database_path: str | Path = Path(".db")):
    """List all vector store configurations in the given directory."""
    database_path = Path(database_path).resolve()
    if not database_path.exists():
        return []

    configs = {}
    for p in database_path.iterdir():
        if (p / "config.json").exists():
            with (p / "config.json").open("r") as f:
                try:
                    config = json.load(f)
                    configs[str(p)] = config
                except json.JSONDecodeError:
                    pass
    return configs

def load_from_path(database):
    """Load a vector store from a given path."""
    if type(database) is str:
        database = Path(database)
    if not database.exists():
        raise ValueError(f"Database path {database} does not exist.")

    if not (database / "config.json").exists():
        raise ValueError(f"No config.json found in {database}. Make sure this is a valid vector store directory.")

    with open(database / "config.json", "r") as f:
        config = json.load(f)

    return Retriever(
        modules=tuple(config["modules"]),
        model=config["model"],
        db_dir=database,
        preprocess=load_plain_theorems
    )

class Retriever:
    """Wrapper around a Chroma vector store."""

    def __init__(self, modules : Tuple[str], model : str, db_dir: str | Path = Path(".db"), preprocess=load_plain_theorems, project_dir: str | Path = Path.cwd()):
        """Initialize a new :class:`Retriever`.

        Parameters
        ----------
        modules : Tuple[str]
            List of Lean modules to index. Cannot be empty.
        model : str
            Name of the embedding model used by the vector store.
        preprocess : Callable
            Callable returning an iterable of documents to index when creating
            a new store.
        db_path : str | Path | None, optional
            Directory where the vector store should be stored.  If ``None`` a
            new directory is created.
        """

        if type(db_dir) is str:
            db_dir = Path(db_dir)
        db_dir = db_dir.resolve()
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        if type(project_dir) is str:
            project_dir = Path(project_dir)
        self.project_dir = project_dir.resolve()
        if not self.project_dir.exists() or not (self.project_dir / "lean-toolchain").exists():
            raise ValueError(f"{self.project_dir} is not a Lean project. Specify a valid Lean project directory using the `project_dir` parameter.")

        modules = tuple(get_all_modules(list(modules), project_dir=self.project_dir))

        self.database_path = None
        for dir, cfg in _list_configs(db_dir).items():
            if cfg["model"] == model and cfg["modules"].sort() == list(modules).sort() and cfg["project_dir"] == self.project_dir.as_posix():
                self.database_path = Path(dir).resolve()
                self.created = cfg["created"]
                break
        if not self.database_path:
            self.database_path = db_dir / f"{time.time()}_{model.lower().replace('/', '_')}"
            self.database_path.mkdir(parents=True, exist_ok=True)
            self.created = False
            self.project_dir = project_dir.resolve()

        self.modules = modules
        self.model = model
        self.preprocess = preprocess
        self._set_config()

        self.vectorstore = None

        if not self.created:
            self.create_vectorstore()
            self.created = True
            self._set_config()


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
            "modules": list(self.modules),
            "created": self.created,
            "project_dir": str(self.project_dir.as_posix()),
        }
        with (self.database_path / "config.json").open("w") as f:
            json.dump(config, f, indent=4)

    def create_vectorstore(self):
        """Create a new Chroma vector store and ingest documents."""

        # for declaration in self.preprocess(list(self.modules), project_dir=self.project_dir):
        #     print("DECL : " + str(declaration))
        #
        # return

        print("Building vectorstore!")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model
        )
        vectorstore = Chroma(
            collection_name=self.model.lower().replace("/", "_") + "_db",
            persist_directory=str(self.database_path),
            embedding_function=embeddings,
        )

        docs_queue = []
        for declaration in self.preprocess(list(self.modules))():
            print(declaration)
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

if __name__ == "__main__":
    retriever = Retriever(
        modules=("Mathlib.Algebra.Group",),
        model="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13",
    )
