"""Test configuration for the data pipeline / evaluation tests (synthetic fixtures only)."""

import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))


def pytest_configure(config):
    # pytest.ini restricts python_files to the Layer 1 test names; these suites use test_*.py.
    config.addinivalue_line("python_files", "test_*.py")
