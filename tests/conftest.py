"""Pytest configuration and shared fixtures.

This module contains pytest configuration and fixtures that are shared
across all test modules.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_current_event_loop() -> None:
    """Keep a main-thread event loop available for legacy sync tests.

    Python 3.13 no longer creates one implicitly for ``asyncio.get_event_loop()``.
    A subset of older unit tests still uses ``run_until_complete`` directly.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            raise RuntimeError("closed event loop")
        return
    except RuntimeError:
        pass

    policy_local = getattr(asyncio.get_event_loop_policy(), "_local", None)
    loop = getattr(policy_local, "_loop", None)
    if loop is None or loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())


_ensure_current_event_loop()


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory path.
    
    Returns:
        Path to the project root directory.
    """
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def ensure_legacy_event_loop() -> None:
    """Provide a current loop for sync tests using ``get_event_loop``."""
    _ensure_current_event_loop()


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """Return the sample documents directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the sample documents directory.
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Return the config directory path.
    
    Args:
        project_root: The project root directory path.
        
    Returns:
        Path to the config directory.
    """
    return project_root / "config"
