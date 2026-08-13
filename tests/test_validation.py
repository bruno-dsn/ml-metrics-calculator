import pandas as pd
import pytest

from src.validation import diagnosticar_distribuicao, inferir_classes, validar_colunas


def test_validacao_rejeita_coluna_ausente():
    with pytest.raises(ValueError):
        validar_colunas(pd.DataFrame({"y_real": [1]}), "Classificação binária")


def test_validacao_regressao_converte_numeros():
    dados = validar_colunas(pd.DataFrame({"y_real": ["1", "2"], "y_previsto": ["1.2", "1.9"]}), "Regressão")
    assert dados["y_real"].dtype.kind in "fi"


def test_inferir_classes_usa_real_e_previsto():
    dados = pd.DataFrame({"y_real": ["a", "b"], "y_previsto": ["a", "c"]})
    assert inferir_classes(dados) == ["a", "b", "c"]


def test_diagnostico_identifica_desbalanceamento():
    resultado = diagnosticar_distribuicao([0] * 80 + [1] * 20)
    assert resultado["desbalanceado"] is True
    assert resultado["proporcao_majoritaria"] == pytest.approx(0.8)
