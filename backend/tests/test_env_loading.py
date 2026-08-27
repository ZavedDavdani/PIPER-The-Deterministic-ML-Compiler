"""
Tests for repository-root .env discovery and loading.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.env import find_project_root, load_project_env, project_env_path
from app.llm import create_llm_provider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider


@pytest.fixture()
def isolated_env_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Minimal repo layout::

        <tmp>/backend/app/.gitkeep
        <tmp>/.env
    """
    app_dir = tmp_path / "backend" / "app"
    app_dir.mkdir(parents=True)
    (app_dir / ".gitkeep").write_text("", encoding="utf-8")
    monkeypatch.setattr("app.env.find_project_root", lambda start=None: tmp_path)
    return tmp_path


def _write_env(root: Path, content: str) -> Path:
    env_path = root / ".env"
    env_path.write_text(content, encoding="utf-8")
    return env_path


class TestProjectRootDiscovery:
    def test_find_project_root_from_backend_package(self):
        root = find_project_root(Path(__file__).resolve().parents[1] / "app" / "env.py")
        assert (root / "backend" / "app").is_dir()
        assert (root / ".env.example").is_file()

    def test_project_env_path_points_at_repository_root(self):
        env_path = project_env_path(Path(__file__).resolve().parents[1] / "app" / "env.py")
        assert env_path.name == ".env"
        assert env_path.parent.name == find_project_root().name


class TestDotenvLoading:
    def test_root_env_is_discovered(self, isolated_env_tree: Path, monkeypatch: pytest.MonkeyPatch):
        env_path = _write_env(
            isolated_env_tree,
            "PIPER_LLM_PROVIDER=openai\nOPENAI_MODEL=gpt-5.6-luna\n",
        )
        monkeypatch.delenv("PIPER_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        loaded = load_project_env()

        assert loaded == env_path
        assert os.environ.get("PIPER_LLM_PROVIDER") == "openai"
        assert os.environ.get("OPENAI_MODEL") == "gpt-5.6-luna"

    def test_openai_api_key_detected_without_exposing_value(
        self, isolated_env_tree: Path, monkeypatch: pytest.MonkeyPatch
    ):
        secret = "sk-test-secret-from-dotenv"
        _write_env(
            isolated_env_tree,
            f"PIPER_LLM_PROVIDER=openai\nOPENAI_API_KEY={secret}\nOPENAI_MODEL=gpt-5.6-luna\n",
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("PIPER_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        load_project_env()

        provider = OpenAIProvider()
        assert bool(provider.api_key) is True
        assert provider.api_key == secret

    def test_provider_factory_selects_openai_from_env(
        self, isolated_env_tree: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_env(
            isolated_env_tree,
            "PIPER_LLM_PROVIDER=openai\nOPENAI_API_KEY=sk-test\nOPENAI_MODEL=gpt-5.6-luna\n",
        )
        monkeypatch.delenv("PIPER_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        load_project_env()

        provider = create_llm_provider()
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-5.6-luna"

    def test_ollama_configuration_still_works_from_env(
        self, isolated_env_tree: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_env(
            isolated_env_tree,
            (
                "PIPER_LLM_PROVIDER=ollama\n"
                "PIPER_OLLAMA_HOST=http://custom-ollama:11434\n"
                "PIPER_LLM_MODEL=custom-model:latest\n"
            ),
        )
        monkeypatch.delenv("PIPER_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("PIPER_OLLAMA_HOST", raising=False)
        monkeypatch.delenv("PIPER_LLM_MODEL", raising=False)

        load_project_env()

        provider = create_llm_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider.host == "http://custom-ollama:11434"
        assert provider.model == "custom-model:latest"

    def test_existing_os_environment_is_not_overwritten_by_dotenv(
        self, isolated_env_tree: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_env(
            isolated_env_tree,
            "PIPER_LLM_PROVIDER=openai\nOPENAI_MODEL=gpt-5.6-luna\n",
        )
        monkeypatch.setenv("PIPER_LLM_PROVIDER", "ollama")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)

        load_project_env()

        assert os.environ.get("PIPER_LLM_PROVIDER") == "ollama"
        assert os.environ.get("OPENAI_MODEL") == "gpt-5.6-luna"
