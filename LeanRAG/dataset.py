from pathlib import Path
import json

from .LeanIO import check_leanRAG_installation, run_lean_command_sync
from .module import Module, Declaration, module_from_path, search_for_lean_file, RestrictedModule, restricted_module_from_path



class Dataset:
    def __init__(self, dataset_json, split="train", name="", lean_dir=Path.cwd(), include_dependencies=False):
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
        self.include_dependencies = False

        # If we want dependencies to be included, we index the current dataset for deps, then add them to a new dataset, and only then enable access to the dependency files through all_files() to prevent an infinite loop
        # TODO: some overlap between dataset and dependencies for dense datasets, combine databases per library?
        if include_dependencies:
            print("indexing dependencies...")
            self._check_install()
            dataset_extended = {self.split: {}}
            for module in self.all_files():
                for decl in module.declarations():
                    for dep in decl.dependencies:
                        # print(dep, type(dep))
                        prefix = dep.module.name.split(".")[0]
                        if prefix not in dataset_extended[self.split]:
                            dataset_extended[self.split][prefix] = []

                        module_path = dep.module.rel_path()
                        has_found = False

                        for existing_dep in dataset_extended[self.split][prefix]:
                            if existing_dep == module_path:
                                has_found = True
                                break
                            elif type(existing_dep) is dict and existing_dep.get("file", "") == module_path:
                                has_found = True
                                if dep.name not in existing_dep["theorems"]:
                                    existing_dep["theorems"].append(dep.name)
                                break

                        if not has_found:
                            dataset_extended[self.split][prefix].append({
                                "file": module_path,
                                "theorems": [dep.name]
                            })

            self.include_dependencies = True
            self.dataset_extended = Dataset(dataset_extended, split=self.split, name=f"{self.name}_dependencies", lean_dir=self.lean_dir, include_dependencies=False)


    def run_lean_command(self, command):
        return run_lean_command_sync(command, project_dir=self.lean_dir)

    def _check_install(self):
        if not self._has_checked_install:
            check_leanRAG_installation(project_dir=self.lean_dir)
        self._has_checked_install = True

    # def children(self):
    #     for k in self.dataset_json[self.split].keys():
    #         yield Module(k, lean_dir=self.lean_dir)

    def all_files(self, ignore_blacklist=False, include_dependencies=True):
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
        if include_dependencies and self.include_dependencies:
            yield from self.dataset_extended.all_files(ignore_blacklist=ignore_blacklist, include_dependencies=False)

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

def dataset_from_json(path, split="train", lean_dir=Path.cwd(), include_dependencies=False):
    """Create a Dataset from a JSON file."""
    if type(path) is str:
        path = Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")
    if not path.is_file():
        raise ValueError(f"Path {path} is not a file.")

    with path.open("r") as f:
        dataset_json = json.load(f)

    return Dataset(dataset_json, split=split, lean_dir=lean_dir, include_dependencies=include_dependencies)

def dataset_from_modules(modules : list[Module], split="train", lean_dir=Path.cwd(),
                         include_dependencies=False):
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

    return Dataset(dataset_json, split=split, lean_dir=lean_dir, include_dependencies=include_dependencies)