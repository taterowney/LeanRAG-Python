from pathlib import Path
from .LeanIO import check_leanRAG_installation, run_lean_command_sync
from .module import Module, Declaration, module_from_path, search_for_lean_file
import json


class RestrictedModule(Module):
    def __init__(self, whitelisted_declarations, dataset : 'Dataset', name="", lean_dir=Path.cwd(), _module_path=None):
        """ whitelisted_declarations should be just literal names with no namespace prefixes, etc. """
        super().__init__(name=name, lean_dir=lean_dir, _module_path=_module_path)
        self.whitelisted_declarations = whitelisted_declarations
        self.dataset = dataset

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

class Dataset:
    def __init__(self, dataset_json, split="train", name="", lean_dir=Path.cwd()):
        self.dataset_json = dataset_json
        self.split = split
        if type(lean_dir) is str:
            lean_dir = Path(lean_dir)
        self.lean_dir = lean_dir.resolve()
        if not ((self.lean_dir / "lakefile.toml").exists() or (self.lean_dir / "lakefile.lean").exists()):
            raise ValueError()
        self._has_checked_install = False
        self.is_file = False
        self.is_dir = False
        if not name:
            self.name = f"dataset_{split}"
        else:
            self.name = name

    def run_lean_command(self, command):
        return run_lean_command_sync(command, project_dir=self.lean_dir)

    def _check_install(self):
        if not self._has_checked_install:
            check_leanRAG_installation(project_dir=self.lean_dir)
        self._has_checked_install = True

    # def children(self):
    #     for k in self.dataset_json[self.split].keys():
    #         yield Module(k, lean_dir=self.lean_dir)

    def all_files(self, ignore_blacklist=False):
        for _, elements in self.dataset_json[self.split].items():
            for elem in elements:
                if type(elem) is str:
                    # yield module_from_path(
                    #     search_for_lean_file(elem, lean_dir=self.lean_dir),
                    #     lean_dir=self.lean_dir
                    # )
                    yield restricted_module_from_path(elem, [], self, lean_dir=self.lean_dir)
                elif type(elem) is dict:
                    yield restricted_module_from_path(elem["file"], [decl.split(".")[-1] for decl in elem["theorems"]], self, lean_dir=self.lean_dir)
                else:
                    raise TypeError(f"Unexpected type {type(elem)} in dataset JSON. Expected str or dict.")

    def path(self):
        return self.lean_dir

    def get_parent(self):
        raise ValueError("Top-level module")

    def get_toplevel(self):
        return self

    def __getitem__(self, item):
        if type(item) is not str:
            raise TypeError("Item must be a string representing the module name.")
        if item in self.dataset_json[self.split]:
            return RestrictedModule([], self, item, lean_dir=self.lean_dir)
        else:
            raise KeyError(f"Module {item} not found in dataset {self.name}.")

    def __eq__(self, other):
        return self.dataset_json == other.dataset_json and self.lean_dir == other.lean_dir

    def __getattr__(self, item):
        return self[item]

def dataset_from_json(path, split="train", lean_dir=Path.cwd()):
    """Create a Dataset from a JSON file."""
    if type(path) is str:
        path = Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")

    with path.open("r") as f:
        dataset_json = json.load(f)

    return Dataset(dataset_json, split=split, lean_dir=lean_dir)

def dataset_from_modules(modules : list[Module], split="train", lean_dir=Path.cwd()):
    """Create a Dataset from a list of modules."""
    if type(modules) is not list:
        raise TypeError("Modules must be a list of Module objects.")

    dataset_json = {split: {}}
    for module in modules:
        top_name = module.get_toplevel().name
        if not isinstance(module, Module):
            raise TypeError(f"Expected Module object, got {type(module)}")
        if top_name not in dataset_json[split]:
            dataset_json[split][top_name] = []
        dataset_json[split][top_name] += [m.path() for m in module.all_files()]

    # remove duplicates
    for k, v in dataset_json[split].items():
        dataset_json[split][k] = list(set(v))

    return Dataset(dataset_json, split=split, lean_dir=lean_dir)