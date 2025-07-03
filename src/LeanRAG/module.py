from pathlib import Path
import json
from LeanIO import check_leanRAG_installation, run_lean_command_sync

SEARCH_PATH_ENTRIES = [
    ".",
    ".lake/packages/*/"
]

class Module:
    def __init__(self, name="", lean_dir=Path.cwd()):
        if type(lean_dir) is str:
            lean_dir = Path(lean_dir)
        self.lean_dir = lean_dir.resolve()
        if not ((self.lean_dir / "lakefile.toml").exists() or (self.lean_dir / "lakefile.lean").exists()):
            raise ValueError()
        self.name = name.split(".")
        # TODO: imported modules
        self.is_file = False
        self.is_dir = False
        for entry in SEARCH_PATH_ENTRIES:
            if entry == ".":
                if (self.lean_dir / ("/".join(self.name) + ".lean")).exists():
                    self.is_file = True
                    self.file_path = self.lean_dir / ("/".join(self.name) + ".lean")
                if (self.lean_dir / "/".join(self.name)).exists():
                    self.is_dir = True
                    self.dir_path = self.lean_dir / ("/".join(self.name))
                if self.is_file and self.is_dir:
                    break
            else:
                for d in self.lean_dir.glob(entry):
                    if (d / ("/".join(self.name) + ".lean")).exists():
                        self.is_file = True
                        self.file_path = d / ("/".join(self.name) + ".lean")
                    if (d / "/".join(self.name)).exists():
                        self.is_dir = True
                        self.dir_path = d / ("/".join(self.name))
                    if self.is_file and self.is_dir:
                        break

        if not self.is_file and not self.is_dir:
            raise ModuleNotFoundError
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
            ret.append(Module(f"{".".join(self.name)}.{file.as_posix().replace(".lean", "").split("/")[-1]}", lean_dir=self.lean_dir))
        return ret

    def all_files(self):
        if not self.is_dir:
            return [self]
        ret = []
        if self.is_file:
            ret.append(self)
        for file in self.dir_path.rglob("*.lean"):
            if file.as_posix().endswith(".lean"):
                ret.append(Module(
                    ".".join(self.name) + "." + file.relative_to(self.dir_path).as_posix().replace(".lean", "").replace("/", "."),
                    lean_dir=self.lean_dir)
                )
        return ret

    def declarations(self, theorems_only=False):
        # Generator to support slow Lean interface
        if not self.is_file:
            raise ModuleNotFoundError
        if not self._has_loaded_theorems:
            self._check_install()
            # TODO: make async, streaming
            for line in self.run_lean_command(f"extract_states {".".join(self.name)}").split("\n"):
                if line:
                    th = json.loads(line)
                    # th["kind"] = "theorem"
                    if theorems_only and th["kind"] != "theorem":
                        continue
                    th = Declaration(th["name"], th["kind"], self, th["decl"], th["initialProofState"])
                    yield th
                    if th not in self._theorems:
                        self._theorems.append(th)
            self._has_loaded_theorems = True
        else:
            for th in self._theorems:
                yield th

    def path(self):
        if self.is_file:
            return self.file_path
        return self.dir_path

    def __getitem__(self, item):
        if type(item) != str:
            raise ValueError
        if not self.is_dir:
            raise ModuleNotFoundError
        for file in self.dir_path.iterdir():
            if file.as_posix().replace(".lean", "").endswith(item.replace("/", "")):
                return Module(f"{".".join(self.name)}.{item}", lean_dir=self.lean_dir)
        raise ModuleNotFoundError

    def __str__(self):
        return ".".join(self.name)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.lean_dir == other.lean_dir and self.name == other.name

    def __getattr__(self, item):
        return self[item]

class Declaration:
    # TODO: maybe include intermediate proof states too? Will that slow down Lean too much?
    def __init__(self, name, kind, module, src, initial_proof_state):
        self.name = name
        self.kind = kind
        self.declaration_name = name.split(".")[-1]
        self.module = module
        self.src = src.strip()
        self.initial_proof_state = initial_proof_state

    def __eq__(self, other):
        return self.name == other.name and self.kind == other.kind and self.module == other.module and self.initial_proof_state == other.initial_proof_state and self.src == other.src

    def __str__(self):
        return self.src

    def __repr__(self):
        return str(self.module) + "." + self.name

if __name__ == "__main__":
    Mathlib = Module("Mathlib", lean_dir="/Users/trowney/Documents/GitHub/test_project")
    # print(list(TestProject.Basic.theorems()))
    print(list(Mathlib.MeasureTheory.PiSystem.declarations()))
