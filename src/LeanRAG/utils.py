import os, re, subprocess, asyncio, multiprocessing, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .annotate import get_goal_annotations


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

def get_all_modules(modules, project_dir=Path.cwd()):
    """
    Get all modules (actual lean files) from a list of module paths that are possibly modules, possibly directories
    """
    out = []
    cmd = "find .lake/packages/ \\( -type f -name '*.lean' -o -type d \\)"
    project_dir = Path(project_dir).resolve()
    lake_packages = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=project_dir
    ).stdout.split("\n")

    lake_packages = {
        "/".join(p.replace(".lake/packages/", "").split("/")[1:]): os.path.join(project_dir, p) for p in lake_packages
    }

    while modules:
        m = modules.pop(0)
        as_path = m.replace(".", "/")
        if os.path.exists(os.path.join(project_dir, as_path + ".lean")):
            out.append(m)
        elif as_path + ".lean" in lake_packages.keys():
            out.append(m)

        # Handle directories
        if os.path.exists(os.path.join(project_dir, as_path)):
            for file in os.listdir(os.path.join(project_dir, as_path)):
                modules.append(m + "." + file.replace(".lean", "").replace("/", "."))
        elif as_path in lake_packages.keys():
            for file in os.listdir(lake_packages[as_path]):
                modules.append(m + "." + file.replace(".lean", "").replace("/", "."))

    paths = list(set(out))
    return paths

def load_plain_theorems(modules, project_dir=Path.cwd()):
    """
    Load theorems from plain text Lean files. Delays evaluation.
    """

    return lambda: load_plain_theorems_(modules, project_dir=project_dir)

def load_plain_theorems_(modules, project_dir=Path.cwd()):
    paths = []
    cmd = "find .lake/packages/ \\( -type f -name '*.lean' -o -type d \\)"
    lake_packages = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=Path(project_dir).resolve()
    ).stdout.split("\n")

    lake_packages = {
        "/".join(p.replace(".lake/packages/", "").split("/")[1:]): os.path.join(os.getcwd(), p) for p in lake_packages
    }
    while modules:
        m = modules.pop(0)
        as_path = m.replace(".", "/")
        if os.path.exists(os.path.join(os.getcwd(), as_path + ".lean")):
            paths.append(os.path.join(os.getcwd(), as_path + ".lean"))
        # elif "./lake/packages/" + as_path + ".lean" in lake_packages.keys():
        elif as_path + ".lean" in lake_packages.keys():
            paths.append(lake_packages[as_path + ".lean"])

        # Handle directories
        if os.path.exists(os.path.join(os.getcwd(), as_path)):
            for file in os.listdir(os.path.join(os.getcwd(), as_path)):
                modules.append(m + "." + file.replace(".lean", "").replace("/", "."))
        elif as_path in lake_packages.keys():
            for file in os.listdir(lake_packages[as_path]):
                modules.append(m + "." + file.replace(".lean", "").replace("/", "."))

    paths = list(set(paths))

    for path in paths:
        assert os.path.exists(path), f"File {path} does not exist. Tate made an oopsie"
        with open(path, "r") as f:
            for decl in get_decls_from_plaintext(f.read()):
                yield decl

def load_annotated_goal_statements(modules, project_dir=Path.cwd()):
    """
    Load theorems from annotated goal state files. Delays evaluation.
    """

    return lambda: load_annotated_goal_state_theorems_(modules, project_dir=project_dir)

def load_annotated_goal_statements_(modules, project_dir=Path.cwd()):
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

def is_theorem(statement):
    decl_re = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)*(theorem|lemma|example|problem|def)\b')
    if decl_re.match(statement):
        return True

def load_annotated_goal_state_theorems(modules, project_dir=Path.cwd()):
    """
    Load theorems from annotated goal state files. Delays evaluation.
    """

    return lambda: load_annotated_goal_state_theorems_(modules, project_dir=project_dir)

def load_annotated_goal_state_theorems_(modules, project_dir=Path.cwd()):
    for statement in load_annotated_goal_statements_(modules, project_dir=project_dir):
        if is_theorem(statement["decl"]):
            yield statement

# if __name__ == "__main__":
#     for decl in load_annotated_goal_state_theorems(["Mathlib.Algebra.Group"], project_dir="../../test_project/"):
#         print(decl)
