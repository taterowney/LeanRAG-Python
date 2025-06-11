from __future__ import annotations
from langchain.globals import set_debug
import time

set_debug(False)

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from concurrent.futures import ThreadPoolExecutor
import os, json, shutil
import subprocess


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def get_library_lean_files(
    library_name="Mathlib",
    path=os.path.join(ROOT_PATH, ".lake", "packages", "mathlib", "Mathlib"),
):
    cmd = f'find {path} -type f -name "*.lean" -print'
    # print(cmd)
    files = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    ).stdout.split("\n")
    print(f"Found {len(files)} files in {library_name} library.")
    for i in range(len(files)):
        try:
            files[i] = (
                (library_name + files[i].split(library_name)[1])
                .replace("/", ".")
                .split(".lean")[0]
            )
        except IndexError:
            files[i] = ""
    return list(filter(None, files))


def save_annotated_library(
    library_name="Mathlib",
    path=os.path.join(ROOT_PATH, ".lake", "packages", "mathlib", "Mathlib"),
    style="new",
):
    modules = get_library_lean_files(library_name, path)
    # print(modules)
    # modules = ['Mathlib.Data.Finset.MulAntidiagonal', 'Mathlib.Data.Complex.Cardinality', 'Mathlib.Topology.Category.TopCat.EffectiveEpi']
    if style == "new":
        if not os.path.exists(
            os.path.join(ROOT_PATH, "RAG", "annotated_new", library_name)
        ):
            os.makedirs(os.path.join(ROOT_PATH, "RAG", "annotated_new", library_name))
    else:
        if not os.path.exists(
            os.path.join(ROOT_PATH, "RAG", "annotated", library_name)
        ):
            os.makedirs(os.path.join(ROOT_PATH, "RAG", "annotated", library_name))

    def annotateModule(module):
        if style == "new":
            cmd = ["lake", "exe", "extract_states", module]
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                universal_newlines=True,
                cwd=ROOT_PATH,
            )
            if out.stdout:
                with open(
                    os.path.join(
                        ROOT_PATH,
                        "RAG",
                        "annotated_new",
                        library_name,
                        f"{module}.jsonl",
                    ),
                    "w",
                ) as f:
                    f.write(out.stdout)
                print(f"Annotated {module}")
            return out.stdout
        else:
            cmd = ["lake", "exe", "StateComments", module]
            out = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                universal_newlines=True,
                cwd=ROOT_PATH,
            )
            with open(
                os.path.join(
                    ROOT_PATH, "RAG", "annotated", library_name, f"{module}.lean"
                ),
                "w",
            ) as f:
                f.write(out.stdout)
            print(f"Annotated {module}")
            return out.stdout

    futures = []
    with ThreadPoolExecutor() as executor:
        for module in modules:
            futures.append(executor.submit(annotateModule, module))
    # for future in futures:
    #     print(future.result())


def create_database_initial_proofstate(
    replace=False, max_docs=None, package_name="Mathlib"
):
    path_to_annotated = os.path.join(ROOT_PATH, "RAG", "annotated_new", package_name)
    if not os.path.exists(path_to_annotated):
        print(
            f"No annotated theorems found at {path_to_annotated}. Run save_annotated_library() to generate them."
        )

    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_initial_proofstate_db"
    )

    if replace:
        if os.path.exists(database_path):
            shutil.rmtree(database_path)

    docs = []
    for file in os.listdir(path_to_annotated):
        if file.endswith(".jsonl"):
            with open(os.path.join(path_to_annotated, file), "r") as f:
                for line in f:
                    j = json.loads(line)
                    # print(j["initialProofState"][:1000])
                    # print("\n\n----------------\n\n")

                    # Only include declarations which are actual proofs
                    used_declarations = ["theorem ", "lemma ", "example ", "problem"]
                    for d in used_declarations:
                        if " "+d+" " in j["decl"] or "\n"+d+" " in j["decl"]:
                            docs.append(
                                Document(
                                    page_content=j["initialProofState"],
                                    metadata={
                                        "decl": j["decl"],
                                        "source": file.replace(".jsonl", ""),
                                    },
                                )
                            )
                            break

    embeddings = HuggingFaceEmbeddings(
        model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    )
    print("Embeddings loaded!")
    vectorstore = Chroma(
        collection_name="Mathlib_initial_proofstate_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )

    docs = docs[:max_docs] if max_docs is not None else docs
    # docs = docs[:100]
    # vectorstore.add_documents(docs)
    chunksize = 1000
    print(f"Chroma loaded, adding {len(docs)} documents...")
    st = time.time()
    num_complete = 0
    for i in range(0, len(docs), chunksize):
        end = min(i + chunksize, len(docs))
        vectorstore.add_documents(docs[i:end])
        num_complete += chunksize
        num_left = len(docs) - num_complete
        total_time = time.time() - st
        time_remaining = total_time * num_left / num_complete
        print(f"Added chunk {i//chunksize}/{len(docs)//chunksize}")
        print(f"    [{round(total_time/60,2)}m | {round(time_remaining / 60,2)} m]")
    # vectorstore.add_documents(docs)
    print("Documents added!")
    return vectorstore


def get_database_retriever(package_name="Mathlib", number_to_retrieve=6, filter={}):
    database_path = os.path.join(
        ROOT_PATH, ".db", f"{package_name.lower()}_initial_proofstate_db"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="hanwenzhu/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-mar13"
    )

    database = Chroma(
        collection_name="Mathlib_initial_proofstate_db",
        persist_directory=database_path,
        embedding_function=embeddings,
    )
    db = database.as_retriever(
        search_type="mmr", search_kwargs={"k": number_to_retrieve}
    )

    return db


def test_average_speed():
    import time

    start = time.time()
    get_database_retriever(replace=True, max_docs=1000)
    print(f"Time per chunk: {(time.time() - start) / 1000}")


def annotate_all_packages(project_home=ROOT_PATH):
    for package_dir in os.listdir(os.path.join(project_home, ".lake", "packages")):
        if os.path.isdir(
            os.path.join(
                ROOT_PATH, ".lake", "packages", package_dir, package_dir.title()
            )
        ):
            save_annotated_library(
                library_name=package_dir,
                path=os.path.join(ROOT_PATH, ".lake", "packages", package_dir),
            )


if __name__ == "__main__":
    # Annotate and save theorems
    # save_annotated_library()

    # Compile the database (takes a hot minute)
    create_database_initial_proofstate(replace=True)

    # Old version
    # create_database_of_annotated(replace=True)

    # Test retrieval
    # IMPORTANT: for new version of retrieval, make sure to run the "getInitialProofState" function in scripts.extract_states, and use that as your query
    # (it will still return the normal text of the theorem)
#     retriever = get_database_retriever()
#     output = retriever.invoke(
#         """α : Type u_1
# E : Type u_2
# inst✝ : NormedField E
# f : α → E
# hf : Multipliable f
# ⊢ Eq (Norm.norm (tprod fun i => f i)) (tprod fun i => Norm.norm (f i))
# """
#     )
#     print(output)
#     print(type(output))
#     for doc in output:
#         print(f"[{doc.metadata}]")
#         print(doc.page_content)
#         print("===============")