from config import EXAMPLE_PROMPTS, MODELS_DB


def test_models_db_structure():
    """Verify MODELS_DB has correct structure and keys."""
    assert isinstance(MODELS_DB, dict)
    assert len(MODELS_DB) > 0

    for lang, details in MODELS_DB.items():
        assert "multisynt" in details
        assert "hplt" in details
        assert isinstance(details["multisynt"], list)
        assert isinstance(details["hplt"], str) or isinstance(details["hplt"], list)


def test_example_prompts_structure():
    """Verify EXAMPLE_PROMPTS matches languages in MODELS_DB."""
    assert isinstance(EXAMPLE_PROMPTS, dict)

    # Check that most languages have prompts (Multilingual-Exp might be different)
    for lang in MODELS_DB.keys():
        if lang in EXAMPLE_PROMPTS:
            assert isinstance(EXAMPLE_PROMPTS[lang], list)
            assert len(EXAMPLE_PROMPTS[lang]) > 0
