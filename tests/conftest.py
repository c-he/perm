"""Shared pytest fixtures and configuration for the PERM project."""

import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration for tests."""
    return {
        "model": {
            "latent_dim": 512,
            "num_layers": 8,
            "hidden_dim": 256
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.002,
            "num_epochs": 100
        },
        "data": {
            "resolution": 512,
            "num_samples": 1000
        }
    }


@pytest.fixture
def sample_hair_data() -> Dict[str, np.ndarray]:
    """Generate sample hair data for testing."""
    np.random.seed(42)  # For reproducible tests
    return {
        "strands": np.random.randn(100, 64, 3).astype(np.float32),
        "roots": np.random.randn(100, 3).astype(np.float32),
        "parameters": np.random.randn(100, 32).astype(np.float32)
    }


@pytest.fixture
def mock_model_weights() -> Dict[str, np.ndarray]:
    """Provide mock model weights for testing."""
    np.random.seed(123)
    return {
        "generator.weight": np.random.randn(512, 256).astype(np.float32),
        "generator.bias": np.random.randn(512).astype(np.float32),
        "discriminator.weight": np.random.randn(256, 512).astype(np.float32),
        "discriminator.bias": np.random.randn(256).astype(np.float32)
    }


@pytest.fixture
def sample_image_data() -> np.ndarray:
    """Generate sample image data for testing."""
    np.random.seed(456)
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up consistent test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # Disable GPU for tests
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def mock_device():
    """Mock device configuration for testing."""
    return "cpu"


@pytest.fixture
def validation_tolerance() -> float:
    """Standard tolerance for numerical comparisons in tests."""
    return 1e-6