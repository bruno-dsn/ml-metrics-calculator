from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_aplicacao_inicia_com_exemplo_binario():
    caminho = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(caminho, default_timeout=20).run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert "Laboratório de métricas" in app.markdown[1].value


def test_aplicacao_abre_classificacao_multiclasse():
    caminho = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(caminho, default_timeout=20).run()
    app.radio[0].set_value("Classificação multiclasse").run()
    assert not app.exception
    assert len(app.tabs) == 3


def test_aplicacao_abre_regressao():
    caminho = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(caminho, default_timeout=20).run()
    app.radio[0].set_value("Regressão").run()
    assert not app.exception
    assert len(app.tabs) == 3
