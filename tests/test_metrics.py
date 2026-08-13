import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    aplicar_limiar,
    curva_limiares,
    matriz_confusao,
    metricas_binarias,
    metricas_multiclasse,
    metricas_regressao,
)


def test_metricas_binarias_caso_conhecido():
    resultado = metricas_binarias([1, 1, 1, 0, 0], [1, 1, 0, 0, 1], 1)
    assert resultado["precisao"] == pytest.approx(2 / 3)
    assert resultado["recall"] == pytest.approx(2 / 3)
    assert resultado["especificidade"] == pytest.approx(0.5)


def test_classe_positiva_pode_ser_texto():
    resultado = metricas_binarias(["não", "sim", "sim"], ["não", "sim", "não"], "sim")
    assert resultado["recall"] == pytest.approx(0.5)


def test_auc_perfeita():
    resultado = metricas_binarias([0, 0, 1, 1], [0, 0, 1, 1], 1, [0.1, 0.2, 0.8, 0.9])
    assert resultado["roc_auc"] == 1
    assert resultado["pr_auc"] == 1


def test_aplicar_limiar():
    resultado = aplicar_limiar([0.2, 0.49, 0.5, 0.9], 0.5, "fraude", "normal")
    assert resultado.tolist() == ["normal", "normal", "fraude", "fraude"]


def test_aplicar_limiar_rejeita_probabilidade_invalida():
    with pytest.raises(ValueError):
        aplicar_limiar([0.2, 1.2], 0.5, 1, 0)


def test_curva_limiares_tem_dezenove_pontos():
    resultado = curva_limiares([0, 0, 1, 1], [0.1, 0.4, 0.7, 0.9], 1)
    assert len(resultado) == 19
    assert {"limiar", "Precisão", "Recall", "F1"} == set(resultado.columns)


def test_metricas_multiclasse_por_classe():
    resumo, por_classe = metricas_multiclasse(["a", "b", "c"], ["a", "a", "c"])
    assert resumo["acuracia"] == pytest.approx(2 / 3)
    assert set(por_classe["classe"]) == {"a", "b", "c"}


def test_regressao_perfeita():
    resultado = metricas_regressao([10, 20, 30], [10, 20, 30])
    assert resultado["mae"] == 0
    assert resultado["rmse"] == 0
    assert resultado["r2"] == 1


def test_regressao_calcula_mape_sem_zeros():
    resultado = metricas_regressao([10, 20], [11, 18])
    assert resultado["mape"] == pytest.approx(0.1)


def test_matriz_confusao_preserva_total():
    matriz = matriz_confusao([0, 0, 1, 1], [0, 1, 1, 1])
    assert matriz.to_numpy().sum() == 4
