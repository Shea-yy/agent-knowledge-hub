"""交付配置回归测试：不需要 Docker daemon，也能锁住可复现性边界。"""

from __future__ import annotations

from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PYTHON_ROOT.parent


def test_runtime_requirements_are_exactly_pinned():
    lines = [
        line.strip()
        for line in (PYTHON_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert lines
    assert all("==" in line for line in lines)
    assert not any(">=" in line or "~=" in line for line in lines)


def test_dockerfile_uses_fixed_runtime_and_non_root_user():
    dockerfile = (PYTHON_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12.13-slim-bookworm" in dockerfile
    assert "USER app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY --chown=app:app . ." in dockerfile


def test_docker_build_context_excludes_secrets_and_mutable_data():
    ignored = (PYTHON_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for pattern in (".env", "uploads/", "table_data/", "agents/*.md", "tests/"):
        assert pattern in ignored


def test_compose_uses_pinned_dependencies_and_health_gates():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "image: neo4j:5.26.30-community" in compose
    assert "image: ghcr.io/chroma-core/chroma:1.5.9" in compose
    assert "condition: service_healthy" in compose
    assert "OPENAI_API_KEY: ${OPENAI_API_KEY:?" in compose
    assert "latest" not in compose


def test_windows_launcher_is_portable_compose_wrapper():
    launcher = (PYTHON_ROOT / "start.bat").read_text(encoding="utf-8")

    assert "docker compose up --build" in launcher
    assert "D:\\agent-knowledge-hub" not in launcher
    assert "taskkill" not in launcher.lower()
    assert ":latest" not in launcher
