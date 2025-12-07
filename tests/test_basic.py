"""
Basic tests to verify witnessed_ph installation and imports.
"""
import pytest


def test_imports():
    """Test that all main modules can be imported."""
    from witnessed_ph import (
        analyse_text_single_slice,
        default_config,
        WitnessedDiagram,
        WitnessedBar,
    )
    assert callable(analyse_text_single_slice)
    assert callable(default_config)


def test_default_config():
    """Test that default config has expected keys."""
    from witnessed_ph import default_config
    
    config = default_config()
    
    assert "embedding_model" in config
    assert "min_persistence" in config
    assert "min_witness_tokens" in config
    assert "lambda_semantic" in config


def test_self_construction_imports():
    """Test Chapter 5 module imports."""
    from witnessed_ph import (
        EventType,
        Journey,
        JourneyGraph,
        SelfMetrics,
        analyse_conversation_from_json,
        compute_self_metrics,
    )
    assert EventType.CARRY.value == "carry"
    assert EventType.RUPTURE_OUT.value == "rupture_out"


@pytest.mark.slow
def test_minimal_analysis():
    """
    Test a minimal analysis (requires model download).
    
    Mark as slow - skip with: pytest -m "not slow"
    """
    from witnessed_ph import analyse_text_single_slice
    
    text = """
    The cat sat on the mat.
    The dog ran in the park.
    """
    
    diagram = analyse_text_single_slice(text, verbose=False)
    
    assert "bars" in diagram
    assert "num_tokens" in diagram
    assert diagram["num_tokens"] > 0
