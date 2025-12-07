"""
Pytest configuration and fixtures for witnessed_ph tests.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (require model downloads)"
    )


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    return {
        "turns": [
            {"role": "user", "content": "Tell me about climate change."},
            {"role": "assistant", "content": "Climate change refers to long-term shifts in temperatures and weather patterns."},
            {"role": "user", "content": "What causes it?"},
            {"role": "assistant", "content": "Human activities, mainly burning fossil fuels, have been the main driver since the 1800s."},
        ]
    }


@pytest.fixture
def sample_text():
    """Sample text for single-slice analysis."""
    return """
    User: I've been thinking about climate change a lot lately.
    Assistant: Climate change is certainly one of the defining challenges of our time.
    User: The economic impacts worry me.
    Assistant: That tension between economic growth and environmental protection is real.
    """
