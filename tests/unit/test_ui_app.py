def test_ui_module_import_does_not_require_optional_streamlit():
    from rag_system.ui.app import main

    assert callable(main)
