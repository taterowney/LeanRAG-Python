import os
import re
import subprocess
import asyncio
import multiprocessing
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .annotate import get_goal_annotations
from .LeanIO import check_leanRAG_installation, run_lean_command, run_lean_command_sync


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


def load_plain_theorems(modules, project_dir: str | Path = Path.cwd()):
    project_dir = Path(project_dir).resolve()
    paths: list[Path] = []
    cmd = r"find .lake/packages/ \( -type f -name '*.lean' -o -type d \)"
    lake_packages = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=project_dir,
    ).stdout.split("\n")

    lake_packages = {
        "/".join(p.replace(".lake/packages/", "").split("/")[1:]): project_dir / p for p in lake_packages
    }
    while modules:
        m = modules.pop(0)
        as_path = m.replace(".", "/")
        candidate = project_dir / f"{as_path}.lean"
        if candidate.exists():
            paths.append(candidate)
        elif as_path + ".lean" in lake_packages:
            paths.append(lake_packages[as_path + ".lean"])

        # Handle directories
        directory = project_dir / as_path
        if directory.exists():
            for file in directory.iterdir():
                modules.append(m + "." + file.name.replace(".lean", "").replace("/", "."))
        elif as_path in lake_packages:
            for file in Path(lake_packages[as_path]).iterdir():
                modules.append(m + "." + file.name.replace(".lean", "").replace("/", "."))

    paths = list(set(paths))

    for path in paths:
        assert Path(path).exists(), f"File {path} does not exist. Tate made an oopsie"
        with Path(path).open("r") as f:
            for decl in get_decls_from_plaintext(f.read()):
                yield decl


def load_annotated_goal_statements(modules, project_dir=Path.cwd()):
    """
    Load theorems from annotated goal state files.
    """

    # semaphore = asyncio.Semaphore(multiprocessing.cpu_count())
    # Doing this without full async for now, I'm lazy
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []

        for module in get_all_modules(modules, project_dir):
            futures.append(
                executor.submit(
                    get_goal_annotations,
                    module,
                    project_dir=project_dir,
                )
            )

        for f in as_completed(futures):
            try:
                for line in f.result().split("\n"):
                    if line.strip():
                        j = json.loads(line.strip())
                        j["page_content"] = j["initialProofState"]
                        j.pop("initialProofState", None)
                        yield j
            except Exception as e:
                print(f"Error processing module: {e}")

def is_theorem(statement: str) -> bool:
    """Return ``True`` if *statement* starts with a Lean declaration."""

    decl_re = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)*(theorem|lemma|example|problem|def)\b')
    if decl_re.match(statement):
        return True
    return False

# def load_annotated_goal_state_theorems(modules, project_dir=Path.cwd()):
#     """
#     Load theorems from annotated goal state files. Delays evaluation.
#     """
#     check_leanRAG_installation(project_dir=project_dir)
#     return lambda: load_annotated_goal_state_theorems_(modules, project_dir=project_dir)

def load_annotated_goal_state_theorems(modules, project_dir: str | Path = Path.cwd()):
    """Yield only theorem declarations from goal state annotations."""
    check_leanRAG_installation(project_dir=project_dir)

    for statement in load_annotated_goal_statements(modules, project_dir=project_dir):
        if is_theorem(statement["decl"]):
            yield statement



def get_all_modules(modules, project_dir: str | Path = Path.cwd()):
    """
    Get all modules (ones that correspond to actual lean files) from a list of module that are possibly lean files or possibly directories
    """
    out = []
    cmd = "find .lake/packages/ \\( -type f -name '*.lean' -o -type d \\)"
    project_dir = Path(project_dir).resolve()
    lake_packages = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=project_dir
    ).stdout.split("\n")

    lake_packages = {
        "/".join(p.replace(".lake/packages/", "").split("/")[1:]): project_dir / p for p in lake_packages
    }

    while modules:
        m = modules.pop(0)
        as_path = m.replace(".", "/")
        if (project_dir / f"{as_path}.lean").exists() or as_path + ".lean" in lake_packages:
            out.append(m)

        # Handle directories
        directory = project_dir / as_path
        if directory.exists():
            for file in directory.iterdir():
                modules.append(m + "." + file.name.replace(".lean", "").replace("/", "."))
        elif as_path in lake_packages:
            for file in Path(lake_packages[as_path]).iterdir():
                modules.append(m + "." + file.name.replace(".lean", "").replace("/", "."))

    out = list(set(out))
    return out


def paths_to_modules(paths, project_dir: str | Path = Path.cwd()):
    """
    Convert a list of file paths to Lean module names
    """
    out = []
    project_dir = Path(project_dir).resolve()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        path = path.resolve()
        if path.suffix != ".lean":
            continue
        p = str(path.relative_to(project_dir))
        try:
            if p.startswith(".lake/packages/") and p.endswith(".lean"):
                p = ".".join(p.replace(".lake/packages/", "").split("/")[1:]).replace(".lean", "")
                out.append(p)
            elif p.endswith(".lean"):
                p = p.replace("/", ".").replace(".lean", "")
                out.append(p)
        except:
            continue
    return out

def modules_to_paths(modules, project_dir: str | Path = Path.cwd()):
    """
    Convert a list of Lean module names to file paths
    """
    out = []
    cmd = "find .lake/packages/ \\( -type f -name '*.lean' -o -type d \\)"
    project_dir = Path(project_dir).resolve()
    lake_packages = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=project_dir
    ).stdout.split("\n")

    lake_packages = {
        "/".join(p.replace(".lake/packages/", "").split("/")[1:]): project_dir / p for p in lake_packages
    }

    while modules:
        m = modules.pop(0)
        as_path = m.replace(".", "/")
        candidate = project_dir / f"{as_path}.lean"
        if candidate.exists():
            out.append(candidate)
        elif as_path + ".lean" in lake_packages:
            out.append(lake_packages[as_path + ".lean"])

        # Handle directories
        directory = project_dir / as_path
        if directory.exists():
            for file in directory.iterdir():
                modules.append(m + "." + file.name.replace(".lean", "").replace("/", "."))
        elif as_path in lake_packages:
            for file in Path(lake_packages[as_path]).iterdir():
                modules.append(m + "." + file.name.replace(".lean", "").replace("/", "."))

    out = list(set(out))
    return out


def get_initial_goal_state(theorem_name: str, module: str, project_dir: str | Path = Path.cwd()) -> str:
    """
    Get the initial goal state for a theorem in a Lean module.

    Parameters
    ----------
    theorem_name : str
        The name of the theorem.
    module : str
        The Lean module where the theorem is defined.
    project_dir : str | Path, default = current working directory
        Path to the Lean workspace root.

    Returns
    -------
    str
        The initial goal state as a string.
    """
    # check_leanRAG_installation(project_dir=project_dir)
    command = f"get_initial_proofstate {module} {theorem_name}"

    return run_lean_command_sync(command, project_dir=project_dir)

def get_relevant_modules(module, project_dir: str | Path = Path.cwd()) -> tuple[str]:
    """
    Get all modules that are relevant to a given module.
    This includes all sub-modules and first-degree imports from other packages.

    Parameters
    ----------
    module : str
        The name of the Lean module.
    project_dir : str | Path, default = current working directory
        Path to the Lean workspace root.

    Returns
    -------
    list[str]
        A list of relevant module names.
    """
    ret = []
    top_dir = Path(project_dir).resolve() / f"{module.replace(".", "/")}"
    top_file = top_dir.with_suffix(".lean")
    if top_dir.exists():
        for file in top_dir.rglob("*.lean"):
            with file.open("r") as f:
                content = f.read()
                for line in content.splitlines():
                    if line.startswith("import "):
                        imported_module = line.split("import ")[1].strip()
                        if imported_module not in ret:
                            ret.append(imported_module)
    elif top_file.exists():
        with top_file.open("r") as f:
            content = f.read()
            for line in content.splitlines():
                if line.startswith("import "):
                    imported_module = line.split("import ")[1].strip()
                    if imported_module not in ret:
                        ret.append(imported_module)
    else:
        raise ValueError(f"Module {module} not found in project directory {project_dir}.")
    ret.extend(get_all_modules(ret, project_dir=project_dir))
    return tuple(ret)

if __name__ == "__main__":
    # for decl in load_annotated_goal_state_theorems(["Mathlib.Algebra.Group"], project_dir="../../test_project/"):
    #     print(decl)
    # print(paths_to_modules(["../../../test_project/.lake/packages/mathlib/Mathlib/Algebra/Group/Basic.lean", "../../../test_project/TestProject.lean"], project_dir="../../../test_project/"))
    pass