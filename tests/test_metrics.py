"""
tests/test_metrics.py
======================
Testes unitários para o metrics_core.py.

Como rodar (na raiz do projeto):
    pytest -v

O que é um "teste unitário"?
-----------------------------
É um pedacinho de código que chama UMA função com uma entrada que a
gente já sabe qual deveria ser a resposta certa, e confere se a função
devolveu isso mesmo. Serve para garantir que, se você mudar o código
no futuro e cometer um erro, o teste "quebra" (falha) e te avisa antes
que o erro chegue no app de verdade.
"""

import sys
import os

# Garante que o Python encontra o metrics_core.py na raiz do projeto,
# mesmo quando o pytest é executado de dentro da pasta tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from metrics_core import (
    classification_metrics,
    regression_metrics,
    calcular_matriz_confusao,
    verificar_desbalanceamento,
)


# ---------- Classificação binária ----------

def test_classificacao_binaria_acerto_total():
    y_true = [1, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 1]
    resultado = classification_metrics(y_true, y_pred, problem_type="binaria")

    assert resultado["acuracia"] == 1.0
    assert resultado["precisao"] == 1.0
    assert resultado["recall"] == 1.0
    assert resultado["f1_score"] == 1.0


def test_classificacao_binaria_erro_total():
    y_true = [1, 1, 1, 1]
    y_pred = [0, 0, 0, 0]
    resultado = classification_metrics(y_true, y_pred, problem_type="binaria")

    assert resultado["acuracia"] == 0.0
    assert resultado["recall"] == 0.0


def test_classificacao_binaria_caso_conhecido():
    # Caso calculado manualmente:
    # y_true: 1 1 1 1 0 0
    # y_pred: 1 1 0 0 0 0
    # VP=2, FN=2, VN=2, FP=0
    # acuracia = 4/6 = 0.6667
    # precisao = VP/(VP+FP) = 2/2 = 1.0
    # recall   = VP/(VP+FN) = 2/4 = 0.5
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 0]
    resultado = classification_metrics(y_true, y_pred, problem_type="binaria")

    assert resultado["acuracia"] == pytest.approx(4 / 6)
    assert resultado["precisao"] == pytest.approx(1.0)
    assert resultado["recall"] == pytest.approx(0.5)
    assert resultado["f1_score"] == pytest.approx(2 / 3, rel=1e-3)


def test_matriz_confusao_formato():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    matriz, labels = calcular_matriz_confusao(y_true, y_pred)

    assert labels == [0, 1]
    assert matriz.shape == (2, 2)
    assert matriz.sum() == 4  # soma da matriz = total de exemplos


# ---------- Classificação multiclasse ----------

def test_classificacao_multiclasse_acerto_total():
    y_true = ["gato", "cachorro", "passaro", "gato"]
    y_pred = ["gato", "cachorro", "passaro", "gato"]
    resultado = classification_metrics(y_true, y_pred, problem_type="multiclasse")

    assert resultado["acuracia"] == 1.0
    assert resultado["f1_score"] == 1.0


def test_classificacao_multiclasse_parcial():
    y_true = ["gato", "cachorro", "passaro"]
    y_pred = ["gato", "gato", "passaro"]
    resultado = classification_metrics(y_true, y_pred, problem_type="multiclasse")

    assert resultado["acuracia"] == pytest.approx(2 / 3)


# ---------- Regressão ----------

def test_regressao_previsao_perfeita():
    y_true = [10, 20, 30, 40]
    y_pred = [10, 20, 30, 40]
    resultado = regression_metrics(y_true, y_pred)

    assert resultado["mae"] == 0.0
    assert resultado["rmse"] == 0.0
    assert resultado["r2"] == pytest.approx(1.0)


def test_regressao_erro_conhecido():
    # erro constante de 2 em cada previsão
    y_true = [10, 20, 30]
    y_pred = [12, 22, 32]
    resultado = regression_metrics(y_true, y_pred)

    assert resultado["mae"] == pytest.approx(2.0)
    assert resultado["rmse"] == pytest.approx(2.0)


def test_regressao_pior_que_a_media_da_r2_negativo():
    # Um "modelo" que sempre chuta um valor bem longe da realidade
    # deve ter R² negativo.
    y_true = [10, 20, 30, 40, 50]
    y_pred = [1000, 1000, 1000, 1000, 1000]
    resultado = regression_metrics(y_true, y_pred)

    assert resultado["r2"] < 0


# ---------- Desbalanceamento ----------

def test_dataset_desbalanceado():
    y_true = [0] * 90 + [1] * 10  # 90% classe 0
    resultado = verificar_desbalanceamento(y_true)

    assert resultado["desbalanceado"] is True
    assert resultado["classe_majoritaria"] == "0"


def test_dataset_balanceado():
    y_true = [0] * 50 + [1] * 50  # 50/50
    resultado = verificar_desbalanceamento(y_true)

    assert resultado["desbalanceado"] is False
