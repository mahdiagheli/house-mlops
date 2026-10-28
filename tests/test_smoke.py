from houseprice_mlops import PROJECT_NAME, __version__


def test_project_metadata() -> None:
    assert PROJECT_NAME == "house-mlops"
    assert __version__ == "0.1.0"