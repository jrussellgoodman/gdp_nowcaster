"""
Shared pytest configuration and fixtures.

Registers the 'integration' marker so pytest doesn't warn about unknown markers.
Integration tests require a live FRED API connection and are automatically
skipped if FRED_API_KEY is not set in the environment.
"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires live FRED API (skipped if FRED_API_KEY is not set)",
    )


@pytest.fixture(scope="session")
def fred_api_key():
    """Provide the FRED API key; skip the test if it is absent."""
    key = os.getenv("FRED_API_KEY")
    if not key:
        pytest.skip("FRED_API_KEY not set — skipping integration test")
    return key
