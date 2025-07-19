import string
from pathlib import Path
import json, re, os, warnings, time, multiprocessing
import duckdb, tempfile, pandas as pd, itertools
from typing import Union, Sequence, Mapping, Any, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from .LeanIO import check_leanRAG_installation, run_lean_command_sync
from .parser import Delimiters, Any, Literal, Word, Repetition, Chars

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
SKIP_CACHE_REBUILD = False


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
                raise ModuleNotFoundError(f"Module {self.name} not found in {self.lean_dir}.")
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
        # if theorems_only:
        #     pattern = re.compile(r'\b(?:theorem|lemma|problem)\s+(\S+)(?=\s|$)')
        # else:
        #     pattern = re.compile(r'\b(?:theorem|lemma|problem|def)\s+(\S+)(?=\s|$)')
        # matches = pattern.findall(strip_lean_comments(self.raw()))
        for data in get_decls_from_plaintext(self.raw(), self.name):
            name = data["name"]
            yield Declaration(name.split(".")[-1], self) #TODO: overlaps if not using full name (see `Mathlib.Topology.Compactness.Lindelof`)

    def path(self):
        if self.is_file:
            return self.file_path
        return self.dir_path

    def rel_path(self):
        if self.is_file:
            return "/".join(self.name_path) + ".lean"
        return "/".join(self.name_path)

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



Row = Union[Sequence[Any], Mapping[str, Any]]


class BatchedRetrievalOperation:
    con = duckdb.connect(".db/cache.duckdb")
    """
    Caches expensive query → rows computations in DuckDB.

    Parameters
    ----------
    db_path : str | None
        Path to the DuckDB file.  Pass ':memory:' (or None) for an in‑memory DB.
    result_table : str
        Name of the table that will hold the generated rows.
    row_generator : Callable[[str], Iterable[Row]]
        A function (or generator) that yields rows **for a single query
        string**.  Each yielded row can be a sequence (tuple / list) or a
        mapping (dict‑like) whose keys are column names.
    processed_table : str
        Name of the bookkeeping table that records which queries were already
        processed.
    """

    def __init__(self, operation) -> None:
        # self.name = operation.__name__
        # assert self.name != "<lambda>", "Might want to name your function something more descriptive"
        # self.con = duckdb.connect(f".db/{self.name}_cache.duckdb")
        self.result_table = "data"
        self.processed_table = "has_processed"
        self.operation = operation

        # Ensure bookkeeping table exists
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.processed_table} (
                module TEXT PRIMARY KEY, attribute TEXT
            )
            """
        )
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.result_table} (
                module TEXT,
                name   TEXT,
                PRIMARY KEY (module, name)
            )
            """
        )
        self._result_table_ready = True
    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def __call__(self, module : Module, decl_name, attr, *args, **kwargs):
        assert attr not in ["module", "name"], "Attributes 'module' and 'name' are reserved and cannot be used."
        top = module.get_toplevel()
        # print("ALL MODULES:", [f.name for f in top.all_files()])
        # if not self._already_processed(module.name, [attr]):
        #     self._populate(top, *args, **kwargs)

        decl_name = _duckdb_escape(decl_name)
        res = self.con.execute(f"SELECT * FROM {self.result_table} WHERE module = '{module.name}' AND name = '{decl_name}'").fetchall()
        if not res or attr not in self.con.table(self.result_table).columns:
            if not SKIP_CACHE_REBUILD:
                self._populate(top, *args, **kwargs)
                res = self.con.execute(
                    f"SELECT * FROM {self.result_table} WHERE module = '{module.name}' AND name = '{decl_name}'").fetchall()
            if not res:
                warnings.warn(f"Declaration {decl_name} in module {module.name} not found in database for {top.name}.")
                return {"kind": "", "src": ""}
        if len(res) > 1:
            if LOG:
                warnings.warn(f"Multiple declarations found for {decl_name} in module {module.name} in database for {top.name}. Returning the first one.")
        res = res[0]

        columns = self.con.table(self.result_table).columns
        res = {col: res[i] for i, col in enumerate(columns)}
        del res["module"]
        del res["name"]

        if attr not in res:
            warnings.warn(f"Attribute '{attr}' not found in declaration {decl_name} in module {module.name} in database for {top.name}.")

        return res

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    # def _already_processed(
    #         self,
    #         query: str,
    #         attributes: Iterable[str],
    # ) -> bool:
    #     """
    #     Given *query* and an iterable of *attributes*, return the set of
    #     attributes that have **not** yet been marked as processed.
    #     """
    #     attr_list = list(attributes)
    #     if not attr_list:
    #         return False
    #
    #     placeholders = ", ".join("?" * len(attr_list))
    #     rows = self.con.execute(
    #         f"""
    #         SELECT attribute
    #         FROM {self.processed_table}
    #         WHERE module = ? AND attribute IN ({placeholders})
    #         """,
    #         [query, *attr_list],
    #     ).fetchall()
    #     print(rows)
    #
    #     processed = {r[0] for r in rows}
    #     return bool(set(attr_list) - processed)

    # def _populate(self, top : Module, *args, **kwargs) -> None:
    #     print(f"No cache found for {self.operation.__name__}, creating one...")
    #     gen = self.operation(top.all_files(), *args, **kwargs)
    #     batch_size = 1000
    #     processed_modules = []
    #
    #     # ------------------------------------------------------------------
    #     # 1) Get a generator and peek at its first row (if any)
    #     # ------------------------------------------------------------------
    #     try:
    #         first_row = next(gen)
    #     except StopIteration:
    #         return  # generator produced nothing
    #
    #     # ------------------------------------------------------------------
    #     # 2) Ensure the table exists *and* has all required columns
    #     # ------------------------------------------------------------------
    #     self._ensure_result_table(first_row)
    #
    #     processed_modules.append(first_row["module"])
    #
    #     # Current column order in DuckDB (used for every INSERT)
    #     col_order = [
    #         col[0]
    #         for col in self.con.execute(f"DESCRIBE {self.result_table}").fetchall()
    #     ]
    #     self._mark_processed(first_row["module"], col_order)
    #
    #
    #     placeholders = ", ".join("?" * len(col_order))
    #     insert_sql = (
    #         f"INSERT OR REPLACE INTO {self.result_table} "
    #         f"({', '.join(col_order)}) VALUES ({placeholders})"
    #     )
    #
    #     def row_to_tuple(r: Mapping[str, object]) -> list[object]:
    #         # Map sparse dict → full ordered list, missing → NULL
    #         return [r.get(c, None) for c in col_order]
    #
    #     # ------------------------------------------------------------------
    #     # 3) Stream rows in batches
    #     # ------------------------------------------------------------------
    #     batch: list[list[object]] = [row_to_tuple(first_row)]
    #     for row in gen:
    #         assert row["name"] and row["module"]
    #         batch.append(row_to_tuple(row))
    #         processed_modules.append(row["module"])
    #         self._mark_processed(row["module"], col_order)
    #         if len(batch) >= batch_size:
    #             self.con.executemany(insert_sql, batch)
    #             batch.clear()
    #
    #     if batch:  # final partial batch
    #         self.con.executemany(insert_sql, batch)
    def _populate(self, top: Module, *args, **kwargs) -> None:
        print(f"No cache found for {self.operation.__name__}, creating one...")
        gen = self.operation(top.all_files(), *args, **kwargs)
        batch_size = 1_000

        # ------------------------------------------------------------------ #
        # 1) first row → table setup
        # ------------------------------------------------------------------ #
        try:
            first_row = next(gen)
        except StopIteration:
            return

        self._ensure_result_table(first_row)

        # complete, *ordered* list of columns that exist in the table
        col_order = [r[0] for r in
                     self.con.execute(f"DESCRIBE {self.result_table}").fetchall()]

        # ------------------------------------------------------------------ #
        # 2) build an UPSERT that preserves old values
        # ------------------------------------------------------------------ #
        pk_cols = ["module", "name"]  # composite PK
        non_pk = [c for c in col_order if c not in pk_cols]

        placeholders = ", ".join("?" * len(col_order))
        update_clause = ", ".join(
            f"{c} = COALESCE(EXCLUDED.{c}, {self.result_table}.{c})"
            for c in non_pk
        )

        insert_sql = (
            f"INSERT INTO {self.result_table} "
            f"({', '.join(col_order)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {update_clause}"
        )

        # ------------------------------------------------------------------ #
        # 3) helper to expand sparse dict → full row
        # ------------------------------------------------------------------ #
        def row_to_tuple(r: Mapping[str, object]) -> list[object]:
            return [r.get(c, None) for c in col_order]

        # ------------------------------------------------------------------ #
        # 4) stream in batches
        # ------------------------------------------------------------------ #
        batch: list[list[object]] = [row_to_tuple(first_row)]
        for row in gen:
            assert row["name"] and row["module"]
            batch.append(row_to_tuple(row))
            if len(batch) >= batch_size:
                self.con.executemany(insert_sql, batch)
                batch.clear()

        if batch:
            self.con.executemany(insert_sql, batch)

    # ----------------------------------------------------------------------
    # Helper: create table (once) or evolve it when new columns appear
    # ----------------------------------------------------------------------
    def _ensure_result_table(self, sample_row: Mapping[str, object]) -> None:
        assert "module" in sample_row and "name" in sample_row, "Sample row must contain 'module' and 'name' keys."
        if not self._result_table_ready:
            cols_def = ", ".join(
                f"{name} {self._duck_type(val)}" for name, val in sample_row.items()
            )
            # self.con.execute(
            #     f"CREATE TABLE IF NOT EXISTS {self.result_table} ({cols_def})"
            # )
            self.con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.result_table} (
                    module TEXT,
                    name   TEXT,
                    PRIMARY KEY (module, name)
                )
                """
            )
            self._result_table_ready = True
        existing_cols = {
            col[0]
            for col in self.con.execute(f"DESCRIBE {self.result_table}").fetchall()
        }
        for name, val in sample_row.items():
            if name not in existing_cols:
                self.con.execute(
                    f"ALTER TABLE {self.result_table} "
                    f"ADD COLUMN {name} {self._duck_type(val)}"
                )


    # def _mark_processed(self, query: str) -> None:
    #     self.con.execute(
    #         f"INSERT OR IGNORE INTO {self.processed_table} VALUES (?)",
    #         [query],
    #     )
    def _mark_processed(self, query: str, attributes: Iterable[str]) -> None:
        """
        Remember that each attribute in *attributes* has been processed for *query*.

        Duplicates are ignored thanks to INSERT OR IGNORE + the PK.
        """
        data = [(query, attr) for attr in attributes]
        self.con.executemany(  # bulk insert
            f"INSERT OR IGNORE INTO {self.processed_table} VALUES (?, ?)",
            data,
        )

    @staticmethod
    def _duck_type(value: Any) -> str:
        """Very small helper to map Python types to DuckDB types."""
        match value:
            case int():
                return "BIGINT"
            case float():
                return "DOUBLE"
            case bool():
                return "BOOLEAN"
            case _:
                return "TEXT"








def _duckdb_escape(s):
    """
    Escape a string for use in a DuckDB query.
    """
    return s.replace("'", "''")

# connects to a single database
# When given a query, it checks if that is in the "has processed" table; if not, it populates the database using the operation and the query's toplevel module

# class BatchedRetrievalOperation:
#     def __init__(self, operation):
#         self.operation = operation
#         self.con = None
#         self.name = operation.__name__
#         assert self.name != "<lambda>", "Might want to name your function something more descriptive"
#         #TODO: when creating multiple databases (i.e. with Dataset), it always uses the first one because self.con is already attached, and looks for things that weren't indexed the first time around
#
#     def check_table_existence(self, top : Module, schema="main"):
#         if self.con is None:
#             if not Path(".db").exists():
#                 os.makedirs(".db")
#             self.con = duckdb.connect(Path(f".db/{top.name}_cache.duckdb").resolve())
#         query = """
#                 SELECT EXISTS (
#                     SELECT 1
#                     FROM information_schema.tables
#                     WHERE table_schema = ? AND table_name = ?
#                 )
#             """
#         return self.con.execute(query, [schema, self.name]).fetchone()[0]
#
#     def populate(self, top : Module, *args, **kwargs):
#         # TODO: comments for whether it has been populated, etc.
#         if not self.check_table_existence(top):
#             assert self.con
#             print(f"No cache found for {self.name} in {top.name}, creating one...")
#
#             it = self.operation(top.all_files(), *args, **kwargs)
#             first = it.__next__()
#             try:
#                 if type(first) == dict:
#                     cols = list(first.keys())
#                     assert "module" in cols and "name" in cols, "Item returned by the function must contain 'module' and 'name' keys."
#                     preprocess = lambda *args: args[0]
#                 else:
#                     raise ValueError(
#                         "Item returned by the function must be a dictionary."
#                     )
#             except Exception as e:
#                 raise ValueError(f"Error processing first item: {e}")
#
#             batch_size = 100
#
#             def write_parquet_pandas(gen, file_path):
#                 first_chunk = True
#                 while True:
#                     chunk = list(
#                         itertools.islice(
#                             map(preprocess, gen),
#                             batch_size
#                         )
#                     )
#                     if not chunk:  # generator exhausted
#                         break
#                     df = pd.DataFrame(chunk, columns=cols)
#                     df.to_parquet(
#                         file_path,
#                         engine="fastparquet",  # ← append needs this
#                         compression="snappy",
#                         index=False,
#                         append=not first_chunk  # append after the first
#                     )
#                     first_chunk = False
#
#             with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
#                 pq_path = tmp.name
#
#             write_parquet_pandas(itertools.chain(iter([first]), it), pq_path)
#
#             self.con.execute(f"DROP TABLE IF EXISTS {self.name}")
#             self.con.execute(f"CREATE TABLE {self.name} AS SELECT * FROM read_parquet('{pq_path}')")
#
#             os.remove(pq_path)
#
#     def __call__(self, module : Module, decl_name, populate = True, *args, **kwargs):
#         top = module.get_toplevel()
#         if populate:
#             self.populate(top, *args, **kwargs)
#         else:
#             if not self.check_table_existence(top):
#                 return None
#         assert self.con is not None
#
#         decl_name = _duckdb_escape(decl_name)
#         res = self.con.execute(f"SELECT * FROM {self.name} WHERE module = '{module.name}' AND name = '{decl_name}'").fetchall()
#         columns = self.con.table(self.name).columns
#         if not res:
#             warnings.warn(f"Declaration {decl_name} in module {module.name} not found in database for {self.name} in {top.name}.")
#             return {"kind": "", "src": ""}
#         if len(res) > 1:
#             if LOG:
#                 warnings.warn(f"Multiple declarations found for {decl_name} in module {module.name} in database for {self.name} in {top.name}. Returning the first one.")
#         res = res[0]
#         res = {col: res[i] for i, col in enumerate(columns)}
#         del res["module"]
#         del res["name"]
#         return res


def get_decls_from_plaintext(raw : str, module_name: str):
    t1 = time.time()
    raw = strip_lean_comments_with_bad_keywords(raw)

    # An attempt at making messy declarations conform to better styling :( its not perfect
    raw_reformatted = ""
    r = raw.split("\n")
    ret = []
    for i, line in enumerate(r):
        if line.startswith("private") or line.startswith("protected") or " private " in line or " protected " in line:
            raw_reformatted += "dummy declaration\n"
            continue
        if "theorem " in line or "lemma " in line or "problem " in line or "def " in line:
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
        if src.strip() and name not in string.punctuation:
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
    modules = list(modules) # ig its fine for now, how much memory could this take anyway
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
                        data["dependencies"] = str(data["dependencies"])
                        yield data
                if res.split("\n")[-1].strip() and LOG:
                    print(f"Processed {json.loads(res.split("\n")[-1].strip())["module"]}")
            except Exception as e:
                warnings.warn(f"Error processing module {future}: {e}")

# TODO:
# - Streaming to batch inference

@BatchedRetrievalOperation
def informalized_theorems(modules : list[Module] | list['RestrictedModule']):
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
            warnings.warn(f"Failed to informalize theorem {thm.name} in module {thm.module.name}.")
        yield {"name": thm.name, "module": thm.module.name, "informal_statement": statement, "informal_proof": proof}

class RestrictedModule(Module):
    def __init__(self, whitelisted_declarations, dataset : 'Dataset', name="", lean_dir=Path.cwd(), _module_path=None):
        """ whitelisted_declarations should be just literal names with no namespace prefixes, etc. """
        super().__init__(name=name, lean_dir=lean_dir, _module_path=_module_path)
        self.whitelisted_declarations = whitelisted_declarations
        self.dataset = dataset
        assert self.dataset is not None

    def declarations(self, theorems_only=False):
        for decl in super().declarations(theorems_only=theorems_only):
            if decl.name in self.whitelisted_declarations or not self.whitelisted_declarations:
                yield decl

    def get_toplevel(self):
        return self.dataset

    def get_parent(self):
        if len(self.name_path) == 1:
            return self.dataset
        parent_name = ".".join(self.name_path[:-1])
        return RestrictedModule([], self.dataset, parent_name, lean_dir=self.lean_dir)

    def __getitem__(self, item):
        raise NotImplementedError #TODO

def restricted_module_from_path(path, whitelisted_declarations : list[str], dataset, lean_dir=Path.cwd()):
    """Create a RestrictedModule from a path. If a path is given, it will be checked for existence; if a string/relative path is given, will search for the Lean file in the project directory."""
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
    return RestrictedModule(whitelisted_declarations, dataset, lean_dir=lean_root, _module_path=path)

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
        if item in ["name", "module"]:
            return object.__getattribute__(self, item)

        elif item in ["kind", "src"]:
            if object.__getattribute__(self, item) is None:
                # Try first from the more CPU-intensive but better-quality "symbolic data" database...
                # attrs = symbolic_theorem_data(self.module, self.name, populate=False)
                # if attrs is None:
                #     # If that doesn't work, extract by processing plaintext instead
                attrs = plaintext_declarations(self.module, self.name, item)
                if "kind" not in attrs or "src" not in attrs:
                    warnings.warn(f"Declaration {self.name} in module {self.module.name} does not have all attributes. Some may be missing.")
                    return ""
                for attr in attrs:
                    if attr != "dependencies":
                        object.__setattr__(self, attr, attrs[attr])
            return object.__getattribute__(self, item)

        elif item in ["initial_proof_state", "dependencies", "full_name"]:
            if object.__getattribute__(self, item) is None:
                # TODO: prefer sourcing "kind" and "src" from the actual script
                attrs = symbolic_theorem_data(self.module, self.name, item)
                # assert "initial_proof_state" in attrs and "dependencies" in attrs and "full_name" in attrs, "Should never print"
                if "initial_proof_state" not in attrs or "dependencies" not in attrs or "full_name" not in attrs:
                    warnings.warn(f"Declaration {self.name} in module {self.module.name} does not have all attributes. Some may be missing.")
                    if item == "dependencies":
                        return []
                    else:
                        return ""

                for attr in attrs:
                    if attr == "dependencies":
                        # Convert json form of dependencies to actual Declarations
                        try:
                            deps = []
                            # raw = attrs[attr].replace("'", '"')
                            # print(attrs["dependencies"].replace("''", "ESCAPEDQUOTE").replace("'", '"').replace("ESCAPEDQUOTE", "'"))
                            # print(type(attrs["dependencies"].replace("'", '"')))
                            # for a in json.loads(attrs["dependencies"].replace("''", "ESCAPEDQUOTE").replace("'", '"').replace("ESCAPEDQUOTE", "'")):
                            for a in eval(attrs["dependencies"]):
                                if a["module"].startswith("Init") or a["module"].startswith("Lean"):
                                    # Skip dependencies from the Lean core
                                    # TODO: is there a way around this without just ignoring them?
                                    continue
                                try:
                                    deps.append(
                                        Declaration(
                                            a["name"].split(".")[-1],  # Get the last part of the name, i.e. without namespaces
                                            # Module(
                                            #     a["module"],
                                            #     lean_dir=self.module.lean_dir
                                            # )
                                            RestrictedModule(
                                                [a["name"].split(".")[-1]],
                                                self.module.get_toplevel(),
                                                name=a["module"],
                                                lean_dir=self.module.lean_dir
                                            )
                                        )
                                    )
                                except ModuleNotFoundError:
                                    pass
                            object.__setattr__(self, "dependencies", deps)
                            # object.__setattr__(self, "dependencies", [Declaration(a["name"], Module(a["module"], lean_dir=self.module.lean_dir)) for a in json.loads(attrs[attr]))
                        except Exception as e:
                            print(attrs)
                            raise ValueError(f"Error processing dependencies for {self.name} in {self.module.name}: {e}")
                    else:
                        object.__setattr__(self, attr, attrs[attr])
            return object.__getattribute__(self, item)

        elif item in ["informal_statement", "informal_proof"]:
            if object.__getattribute__(self, item) is None:
                attrs = informalized_theorems(self.module, self.name, item)
                # assert "informal_statement" in attrs and "informal_proof" in attrs, f"the data for {self.name} was not found in the database for {self.module.name}"
                if "informal_statement" not in attrs or "informal_proof" not in attrs:
                    warnings.warn(f"Declaration {self.name} in module {self.module.name} does not have all attributes. Some may be missing.")
                    return ""
                for attr in attrs:
                    if attr != "dependencies":
                        object.__setattr__(self, attr, attrs[attr])
            return object.__getattribute__(self, item)


if __name__ == "__main__":
    # Mathlib = Module("Mathlib", lean_dir="/Users/trowney/Documents/GitHub/test_project")

    # for decl in Mathlib.Algebra.AddConstMap.Basic.declarations():
    #     print(decl.initial_proof_state)
    #     print("\n\n\n")
    pass