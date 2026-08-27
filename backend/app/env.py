"""
Centralized .env loading for the PIPER backend.

Loads the repository-root ``.env`` file (sibling of the ``backend/``
package) based on this module's location — independent of the process
working directory.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_ROOT_MARKER = "backend"


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the PIPER repository root by walking upward until
    ``backend/app/`` is found.
    """
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / _ROOT_MARKER / "app").is_dir():
            return candidate
    raise RuntimeError("Could not locate PIPER project root (expected backend/app/).")


def project_env_path(start: Path | None = None) -> Path:
    """Absolute path to the repository-root ``.env`` file."""
    return find_project_root(start) / ".env"


def load_project_env(*, override: bool = False) -> Path | None:
    """
    Load the repository-root ``.env`` into ``os.environ``.

    Existing process environment variables take precedence unless
    ``override=True`` (python-dotenv's default is ``override=False``).
    """
    env_path = project_env_path()
    if env_path.is_file():
        load_dotenv(env_path, override=override)
        return env_path
    return None
