from pathlib import Path
import json, re, os, warnings, time, multiprocessing
import duckdb, tempfile, pandas as pd, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from .LeanIO import check_leanRAG_installation, run_lean_command_sync
from .parserv4 import Delimiters, Any, Literal, Word, Repetition, Chars

block_comments_pat = Delimiters("/-", Any(), "-/").remove()
inline_comments_pat = (("--" + Any()).remove() + "\n")
theorems_pat = ((Literal("theorem") | "lemma" | "problem" | "def").extract("kind") + Word().extract("name") + Any() + ":=" + Any() + "\n" + Repetition(Literal("  ") + Any() + "\n", min_num=0) + "\n").extract("src")
block_comments_bad_keywords_pat = Delimiters("/-", Any() + (Literal("theorem") | "lemma" | "problem" | "def" | ":=") + Any(), "-/").remove()
inline_comments_bad_keywords_pat = (("--" + Chars(lambda x: x != "\n") + (Literal("theorem") | "lemma" | "problem" | "def" | ":=") + Chars(lambda x: x != "\n")).remove() + "\n")
# TODO: theorems_pat doens't work if a theorem is the last thing in a file (???)

SEARCH_PATH_ENTRIES = [
    ".",
    ".lake/packages/*/"
]

LOG = False

class Module:
    def __init__(self, name="", lean_dir=Path.cwd(), _module_path=None):
        if type(lean_dir) is str:
            lean_dir = Path(lean_dir)
        self.lean_dir = lean_dir.resolve()
        if not ((self.lean_dir / "lakefile.toml").exists() or (self.lean_dir / "lakefile.lean").exists()):
            raise ValueError()
        self.name = name
        self.name_path = self.name.split(".")
        # TODO: imported modules

        if self.name:
            self.is_file = False
            self.is_dir = False
            for entry in SEARCH_PATH_ENTRIES:
                if entry == ".":
                    if (self.lean_dir / ("/".join(self.name_path) + ".lean")).exists():
                        self.is_file = True
                        self.file_path = self.lean_dir / ("/".join(self.name_path) + ".lean")
                    if (self.lean_dir / "/".join(self.name_path)).exists():
                        self.is_dir = True
                        self.dir_path = self.lean_dir / ("/".join(self.name_path))
                    if self.is_file or self.is_dir:
                        break
                else:
                    for d in self.lean_dir.glob(entry):
                        if (d / ("/".join(self.name_path) + ".lean")).exists():
                            self.is_file = True
                            self.file_path = d / ("/".join(self.name_path) + ".lean")
                        if (d / "/".join(self.name_path)).exists():

                            self.is_dir = True
                            self.dir_path = d / ("/".join(self.name_path))
                        if self.is_file or self.is_dir:
                            break

            if not self.is_file and not self.is_dir:
                raise ModuleNotFoundError
        elif _module_path is not None:
            if _module_path.suffix == ".lean":
                self.is_file = True
                self.file_path = _module_path
                if Path(_module_path.as_posix().replace(".lean", "")).exists():
                    self.is_dir = True
                    self.dir_path = Path(_module_path.as_posix().replace(".lean", ""))
            elif _module_path.is_dir():
                self.is_dir = True
                self.dir_path = _module_path
                if Path(_module_path.as_posix() + ".lean").exists():
                    self.is_file = True
                    self.file_path = Path(_module_path.as_posix() + ".lean")
            rel_path = _module_path.relative_to(self.lean_dir).as_posix()
            if rel_path.startswith(".lake"):
                rel_path = "/".join(rel_path.split("/")[3:])
            self.name = rel_path.replace(".lean", "").replace("/", ".")

        self._has_checked_install = False
        self._theorems = []
        self._has_loaded_theorems = False

    def run_lean_command(self, command):
        return run_lean_command_sync(command, project_dir=self.lean_dir)

    def _check_install(self):
        if not self._has_checked_install:
            check_leanRAG_installation(project_dir=self.lean_dir)
        self._has_checked_install = True

    def raw(self):
        if not self.is_file:
            raise ValueError
        with open(self.file_path, "r") as f:
            return f.read()

    def children(self):
        if not self.is_dir:
            return []
        ret = []
        for file in self.dir_path.iterdir():
            ret.append(Module(f"{".".join(self.name_path)}.{file.as_posix().replace(".lean", "").split("/")[-1]}", lean_dir=self.lean_dir))
        return ret

    def all_files(self, ignore_blacklist=False):
        if not self.is_dir:
            return [self]
        ret = []
        if self.is_file:
            ret.append(self)
        for file in self.dir_path.rglob("*.lean"):
            if file.as_posix().endswith(".lean"):
                if not ignore_blacklist and self.name_path[0] == "Mathlib" and any(f"Mathlib/{x}" in file.as_posix() for x in ["Condensed", "Control", "Deprecated", "Lean", "Mathport", "Std", "Tactic", "Testing", "Util"]):
                    continue
                ret.append(Module(
                    ".".join(self.name_path) + "." + file.relative_to(self.dir_path).as_posix().replace(".lean", "").replace("/", "."),
                    lean_dir=self.lean_dir)
                )
        return ret

    def relevant_files(self):
        """current module + imports"""
        if not self.is_file:
            return [self]
        ret = [self]
        for line in self.raw().split("\n"):
            if line.startswith("import "):
                ret.extend(Module(line.split("import ")[1].strip(), lean_dir=self.lean_dir).relevant_files())
        return ret

    def declarations(self, theorems_only=False):
        # Generator to support slow Lean interface
        if not self.is_file:
            raise ModuleNotFoundError
        if theorems_only:
            pattern = re.compile(r'\b(?:theorem|lemma|problem)\s+(\S+)(?=\s|$)')
        else:
            pattern = re.compile(r'\b(?:theorem|lemma|problem|def)\s+(\S+)(?=\s|$)')
        matches = pattern.findall(strip_lean_comments(self.raw()))
        for m in matches:
            yield Declaration(m.split(".")[-1], self) #TODO: overlaps if not using full name (see `Mathlib.Topology.Compactness.Lindelof`)

    def path(self):
        if self.is_file:
            return self.file_path
        return self.dir_path

    def get_parent(self):
        if len(self.name_path) == 1:
            raise ModuleNotFoundError("Top-level module")
        parent_name = ".".join(self.name_path[:-1])
        return Module(parent_name, lean_dir=self.lean_dir)

    def get_toplevel(self):
        """Get the top-level module (the one without a parent)."""
        if len(self.name_path) == 1:
            return self
        return Module(self.name_path[0], lean_dir=self.lean_dir)

    def __getitem__(self, item):
        if type(item) != str:
            raise TypeError
        if not self.is_dir:
            raise ModuleNotFoundError
        for file in self.dir_path.iterdir():
            if file.as_posix().replace(".lean", "").endswith(item.replace("/", "")):
                return Module(f"{".".join(self.name_path)}.{item}", lean_dir=self.lean_dir)
        raise ModuleNotFoundError

    def __str__(self):
        return ".".join(self.name_path)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.lean_dir == other.lean_dir and self.name_path == other.name

    def __getattr__(self, item):
        return self[item]

def search_for_lean_file(path_suffix : str, lean_dir: Path=Path.cwd()):
    for entry in SEARCH_PATH_ENTRIES:
        if entry == ".":
            if (lean_dir / path_suffix).exists():
                return lean_dir / path_suffix
        else:
            for d in lean_dir.glob(entry):
                if (d / path_suffix).exists():
                    return d / path_suffix
    raise FileNotFoundError(f"Could not find Lean file {path_suffix} in {lean_dir} or its subdirectories.")

def module_from_path(path: str | Path, lean_dir: str | Path | None = None):
    """Create a Module from a path."""
    if type(path) is str:
        path = search_for_lean_file(path, lean_dir=lean_dir)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")
    if lean_dir is None:
        lean_root = path
        while not (lean_root / "lakefile.toml").exists() and not (lean_root / "lakefile.lean").exists():
            if lean_root.parent == lean_root:
                raise ValueError(f"Could not find Lean project root for path {path}.")
            lean_root = lean_root.parent
    else:
        if type(lean_dir) is str:
            lean_root = Path(lean_dir)
        else:
            lean_root = lean_dir
        if not (lean_root / "lakefile.toml").exists() and not (lean_root / "lakefile.lean").exists():
            raise ValueError(f"{lean_root} is not a Lean project root.")
    return Module(lean_dir=lean_root, _module_path=path)

def get_all_files(modules):
    """
    Get all modules (ones that correspond to actual lean files) from a list of module that are possibly lean files or possibly directories
    """
    return list(set(sum([m.all_files() for m in modules], [])))

def is_theorem(statement: str) -> bool:
    """Return ``True`` if *statement* starts with a Lean declaration."""

    decl_re = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)*(theorem|lemma|problem)\b')
    if decl_re.match(statement):
        return True
    return False

def strip_lean_comments(src: str) -> str:

    t = block_comments_pat.get_removed(src + "\n")
    return inline_comments_pat.get_removed(t)

def strip_lean_comments_with_bad_keywords(src: str) -> str:
    """
    Strip Lean comments that contain keywords like `theorem`, `lemma`, `problem`, or `:=` so they don't screw up the parsing of actual declarations
    """
    t = block_comments_bad_keywords_pat.get_removed(src + "\n")
    return inline_comments_bad_keywords_pat.get_removed(t)




def _duckdb_escape(s):
    """
    Escape a string for use in a DuckDB query.
    """
    return s.replace("'", "''")


# def _table_exists(con, name, schema="main"):
#     """
#     Return True if `schema.name` exists, else False.
#     """
#     query = """
#                     SELECT EXISTS (
#                         SELECT 1
#                         FROM information_schema.tables
#                         WHERE table_schema = ? AND table_name = ?
#                     )
#                 """
#     return con.execute(query, [schema, name]).fetchone()[0]

class BatchedRetrievalOperation:
    def __init__(self, operation):
        self.operation = operation
        self.con = None
        self.name = operation.__name__
        assert self.name != "<lambda>", "Might want to name your function something more descriptive"

    def check_table_existence(self, top : Module, schema="main"):
        if self.con is None:
            if not Path(".db").exists():
                os.makedirs(".db")
            self.con = duckdb.connect(Path(f".db/{top.name}_cache.duckdb").resolve())
        query = """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = ? AND table_name = ?
                )
            """
        return self.con.execute(query, [schema, self.name]).fetchone()[0]

    def populate(self, top : Module, *args, **kwargs):
        # TODO: comments for whether it has been populated, etc.
        if not self.check_table_existence(top):
            assert self.con
            print(f"No cache found for {self.name} in {top.name}, creating one...")

            it = self.operation(top.all_files(), *args, **kwargs)
            first = it.__next__()
            try:
                if type(first) == dict:
                    cols = list(first.keys())
                    assert "module" in cols and "name" in cols, "Item returned by the function must contain 'module' and 'name' keys."
                    preprocess = lambda *args: args[0]
                else:
                    raise ValueError(
                        "Item returned by the function must be a dictionary."
                    )
            except Exception as e:
                raise ValueError(f"Error processing first item: {e}")

            batch_size = 100

            def write_parquet_pandas(gen, file_path):
                first_chunk = True
                while True:
                    chunk = list(
                        itertools.islice(
                            map(preprocess, gen),
                            batch_size
                        )
                    )
                    if not chunk:  # generator exhausted
                        break
                    df = pd.DataFrame(chunk, columns=cols)
                    df.to_parquet(
                        file_path,
                        engine="fastparquet",  # ← append needs this
                        compression="snappy",
                        index=False,
                        append=not first_chunk  # append after the first
                    )
                    first_chunk = False

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                pq_path = tmp.name

            write_parquet_pandas(itertools.chain(iter([first]), it), pq_path)

            self.con.execute(f"DROP TABLE IF EXISTS {self.name}")
            self.con.execute(f"CREATE TABLE {self.name} AS SELECT * FROM read_parquet('{pq_path}')")

            os.remove(pq_path)

    def __call__(self, module : Module, decl_name, populate = True, *args, **kwargs):
        top = module.get_toplevel()
        if populate:
            self.populate(top, *args, **kwargs)
        else:
            if not self.check_table_existence(top):
                return None
        assert self.con is not None

        decl_name = _duckdb_escape(decl_name)
        res = self.con.execute(f"SELECT * FROM {self.name} WHERE module = '{module.name}' AND name = '{decl_name}'").fetchall()
        columns = self.con.table(self.name).columns
        if not res:
            warnings.warn(f"Declaration {decl_name} in module {module.name} not found in database for {self.name} in {top.name}.")
            return {"kind": "", "src": ""}
        if len(res) > 1:
            if LOG:
                warnings.warn(f"Multiple declarations found for {decl_name} in module {module.name} in database for {self.name} in {top.name}. Returning the first one.")
        res = res[0]
        res = {col: res[i] for i, col in enumerate(columns)}
        del res["module"]
        del res["name"]
        return res


def get_decls_from_plaintext(raw : str, module_name: str):
    t1 = time.time()
    raw = strip_lean_comments_with_bad_keywords(raw)

    # An attempt at making messy declarations conform to better styling :( its not perfect
    raw_reformatted = ""
    r = raw.split("\n")
    ret = []
    for i, line in enumerate(r):
        if " theorem " in line or " lemma " in line or " problem " in line or " def " in line:
            line = "\n" + line
        elif i > 0 and not line.startswith("  ") and r[i - 1].startswith("  "):
            line = "\n" + line
        raw_reformatted += line + "\n"
    raw = raw_reformatted.strip() + "\n\n-- dummy comment"

    for kind, name, src in theorems_pat.get_extracted(raw + "\n"):
        if kind in ["lemma", "problem"]:
            kind = "theorem"
        elif kind not in ["theorem", "def"]:
            warnings.warn(f"Unknown kind {kind} in module {module_name}. Skipping.")
            continue
        if src.strip():
            ret.append({"name": name.split(".")[-1], "module": module_name, "kind": kind, "src": src.strip()})
    if LOG: print(f"Processed {module_name} in {time.time() - t1:.2f} seconds")
    return ret

@BatchedRetrievalOperation
def plaintext_declarations(modules : list[Module]):
    """
    Get declarations from a plaintext Lean file.
    This function assumes that declarations are separated by newlines.
    """

    # TODO: ProcessPoolExecutor runs the entire file multiple times, creating problems with multiple connections to the database
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(get_decls_from_plaintext, m.raw(), m.name) for m in modules]
    #     for future in as_completed(futures):
    #         try:
    #             yield from future.result()
    #         except Exception as e:
    #             warnings.warn(f"Error processing module {future}: {e}")
    for m in modules:
        yield from get_decls_from_plaintext(m.raw(), m.name)

@BatchedRetrievalOperation
def symbolic_theorem_data(modules : list[Module]):

    modules[0]._check_install()
    threads = max(multiprocessing.cpu_count() // 2, 1)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(m.run_lean_command, f"extract_declarations_info {m.name}") for m in modules if m.is_file]
        for future in as_completed(futures):
            try:
                res = future.result()
                for line in res.split("\n"):
                    if line.strip():
                        data = json.loads(line)
                        data["full_name"] = data["name"]
                        data["name"] = data["name"].split(".")[-1] # Script outputs name including any namespaces wrapping it, so get rid of these and keep them only in full_name
                if res.split("\n")[-1].strip() and LOG:
                    print(f"Processed {json.loads(res.split("\n")[-1].strip())["module"]}")
            except Exception as e:
                warnings.warn(f"Error processing module {future}: {e}")

# TODO:
# - Streaming to batch inference

@BatchedRetrievalOperation
def informalized_theorems(modules : list[Module]):
    from .agent_boilerplate import Client
    from .informalize_utils import make_prompt, process_response
    client = Client(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        model_source="ray"
    )
    # client = Client(
    #     model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #     model_source="test"
    # )
    theorems = [thm for m in modules for thm in m.declarations()]
    prompts = [make_prompt(thm.src) for thm in theorems]

    responses = client.batch_completion(prompts)

    for res, thm in zip(responses, theorems):
        statement, proof, success = process_response(res.choices[0].message.content)
        # statement, proof, success = res.choices[0].message.content, res.choices[0].message.content, True

        if not success:
            warnings.warn(f"Failed to informalize theorem {res.metadata['name']} in module {res.metadata['module']}.")
        yield {"name": thm.name, "module": thm.module.name, "informal_statement": statement, "informal_proof": proof}



class Declaration:
    # TODO: maybe include intermediate proof states, separate out ProofAsSorry-able stuff from things that need full proof
    def __init__(self, name, module):
        self.name = name
        self.module = module

        self.full_name = None
        self.kind = None
        self.src = None
        self.initial_proof_state = None
        self.dependencies = None
        self.informal_statement = None
        self.informal_proof = None

    def __eq__(self, other):
        return self.name == other.name and self.module == other.module

    def __str__(self):
        return self.src

    def __repr__(self):
        return str(self.module) + "." + self.name

    def __getattribute__(self, item):
        if item in ["name", "declaration_name", "module"]:
            return object.__getattribute__(self, item)

        elif item in ["kind", "src"]:
            if object.__getattribute__(self, item) is None:
                # Try first from the more CPU-intensive but better-quality "symbolic data" database...
                attrs = symbolic_theorem_data(self.module, self.name, populate=False)
                if attrs is None:
                    # If that doesn't work, extract by processing plaintext instead
                    attrs = plaintext_declarations(self.module, self.name)
                assert "kind" in attrs and "src" in attrs, "Should never print"
                for attr in attrs:
                    object.__setattr__(self, attr, attrs[attr])
            return object.__getattribute__(self, item)

        elif item in ["initial_proof_state", "dependencies", "full_name"]:
            if object.__getattribute__(self, item) is None:
                # TODO: prefer sourcing "kind" and "src" from the actual script
                attrs = symbolic_theorem_data(self.module, self.name)
                assert "initial_proof_state" in attrs and "dependencies" in attrs and "full_name" in attrs, "Should never print"
                for attr in attrs:
                    if attr == "dependencies":
                        # Convert json form of dependencies to actual Declarations
                        object.__setattr__(self, "dependencies", [Declaration(a["name"], a["module"]) for a in attrs[attr]])
                    else:
                        object.__setattr__(self, attr, attrs[attr])
            return object.__getattribute__(self, item)

        elif item in ["informal_statement", "informal_proof"]:
            if object.__getattribute__(self, item) is None:
                attrs = informalized_theorems(self.module, self.name)
                print(attrs)
                assert "informal_statement" in attrs and "informal_proof" in attrs, f"the data for {self.name} was not found in the database for {self.module.name}"
                for attr in attrs:
                    object.__setattr__(self, attr, attrs[attr])
            return object.__getattribute__(self, item)


if __name__ == "__main__":
    # Mathlib = Module("Mathlib", lean_dir="/Users/trowney/Documents/GitHub/test_project")

    # for decl in Mathlib.Algebra.AddConstMap.Basic.declarations():
    #     print(decl.initial_proof_state)
    #     print("\n\n\n")
    pass