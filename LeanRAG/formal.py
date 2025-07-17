from .module import Module, get_all_files


def load_annotated_goal_state_theorems(modules):
    """
    Load theorems from annotated goal state files.
    """
    modules = get_all_files(modules)
    for m in modules:
        for d in m.declarations(theorems_only=True):
            yield {
                "page_content": d.initial_proof_state,
                "name": d.declaration_name,
                "kind": d.kind,
                "module": d.module,
            }

def get_initial_goal_state(theorem_name: str, module: Module) -> str:
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

    for thm in module.declarations(theorems_only=True):
        if thm.declaration_name == theorem_name or thm.name == theorem_name:
            return thm.initial_proof_state