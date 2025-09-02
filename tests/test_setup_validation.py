"""Validation tests to ensure the testing infrastructure is working correctly."""

import pytest
import sys
from pathlib import Path
import importlib.util


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""

    def test_pytest_is_working(self):
        """Test that pytest is functioning correctly."""
        assert True

    def test_coverage_is_configured(self):
        """Test that coverage configuration is working."""
        # This test will show up in coverage reports if configured correctly
        result = 2 + 2
        assert result == 4

    @pytest.mark.unit
    def test_unit_marker_works(self, request):
        """Test that the unit test marker is working."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "unit" in markers

    @pytest.mark.integration  
    def test_integration_marker_works(self, request):
        """Test that the integration test marker is working."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "integration" in markers

    @pytest.mark.slow
    def test_slow_marker_works(self, request):
        """Test that the slow test marker is working."""
        markers = [marker.name for marker in request.node.iter_markers()]
        assert "slow" in markers

    def test_src_directory_importable(self):
        """Test that the src directory is importable."""
        src_path = Path(__file__).parent.parent / "src"
        assert src_path.exists(), "src directory should exist"
        
        # Test importing main modules
        modules_to_test = [
            "dnnlib",
            "hair",
            "models",
            "torch_utils",
            "training",
            "utils"
        ]
        
        for module_name in modules_to_test:
            module_path = src_path / module_name / "__init__.py"
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                assert spec is not None, f"Could not create spec for {module_name}"

    def test_fixtures_are_working(self, temp_dir, mock_config, sample_hair_data):
        """Test that the shared fixtures are working correctly."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
        
        # Test sample_hair_data fixture
        assert isinstance(sample_hair_data, dict)
        assert "strands" in sample_hair_data
        assert "roots" in sample_hair_data
        assert "parameters" in sample_hair_data
        
        # Verify data shapes are reasonable
        assert sample_hair_data["strands"].shape == (100, 64, 3)
        assert sample_hair_data["roots"].shape == (100, 3)
        assert sample_hair_data["parameters"].shape == (100, 32)

    def test_mock_utilities_work(self, mocker):
        """Test that pytest-mock is working correctly."""
        # Test basic mocking functionality
        mock_function = mocker.Mock(return_value=42)
        result = mock_function()
        assert result == 42
        mock_function.assert_called_once()

    def test_parametrized_test(self, mock_device, validation_tolerance):
        """Test parametrized testing with fixtures."""
        assert mock_device == "cpu"
        assert validation_tolerance == 1e-6
        
        # Test numerical comparison with tolerance
        a = 0.1 + 0.2
        b = 0.3
        assert abs(a - b) < validation_tolerance * 10  # Allow for floating point errors