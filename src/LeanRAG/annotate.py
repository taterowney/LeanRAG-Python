import asyncio
from pathlib import Path

from .LeanIO import run_lean_command, check_leanRAG_installation

#TODO: make everything else actually work with async
async def get_goal_annotations_async(
    module: str,
    project_dir: str | Path = Path.cwd(),
) -> str:
    """
    Get goal annotations for a Lean module.

    Parameters
    ----------
    module : str
        The name of the Lean module.
    project_dir : str | Path, default = current working directory
        Path to the Lean workspace root.

    Returns
    -------
    str
        The goal annotations as a string.
    """
    project_dir = Path(project_dir).resolve()
    command = f"extract_states {module}"

    try:
        output = await run_lean_command(command, project_dir)
        return output.strip()
    except RuntimeError as e:
        raise RuntimeError(f"Failed to get goal annotations: {e}")


def get_goal_annotations(
    module: str,
    project_dir: str | Path = Path.cwd(),
) -> str:
    """
    Get goal annotations for a Lean module synchronously.

    Parameters
    ----------
    module : str
        The name of the Lean module.
    project_dir : str | Path, default = current working directory
        Path to the Lean workspace root.

    Returns
    -------
    str
        The goal annotations as a string.
    """
    return asyncio.run(get_goal_annotations_async(module, project_dir))