"""
tests/test_explicacoes.py
===========================
Testes unitários para as funções de veredito automático em explicacoes.py.

Aqui não testamos o texto exato (isso mudaria toda hora e quebraria o
teste à toa), testamos o "status" (bom / atencao / ruim), que é o que
decide a cor da caixa mostrada na tela. Isso garante que a regra de
negócio (os limiares de precisão, recall e R²) continua correta.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explicacoes import avaliar_modelo_classificacao, avaliar_modelo_regressao


# ---------- Classificação binária ----------

def test_precisao_alta_recall_baixo_gera_atencao():
    metrics = {"precisao": 0.9, "recall": 0.3, "f1_score": 0.45}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="binaria")
    assert resultado["status"] == "atencao"
    assert "Diagnóstico médico" in resultado["evitar_para"][0] or any(
        "médico" in item for item in resultado["evitar_para"]
    )


def test_recall_alto_precisao_baixa_gera_atencao():
    metrics = {"precisao": 0.3, "recall": 0.9, "f1_score": 0.45}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="binaria")
    assert resultado["status"] == "atencao"
    assert resultado["titulo"] == "Recall alto, precisão baixa"


def test_precisao_e_recall_altos_gera_bom():
    metrics = {"precisao": 0.9, "recall": 0.85, "f1_score": 0.87}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="binaria")
    assert resultado["status"] == "bom"


def test_precisao_e_recall_baixos_gera_ruim():
    metrics = {"precisao": 0.3, "recall": 0.2, "f1_score": 0.24}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="binaria")
    assert resultado["status"] == "ruim"


def test_valores_intermediarios_gera_atencao():
    metrics = {"precisao": 0.65, "recall": 0.6, "f1_score": 0.62}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="binaria")
    assert resultado["status"] == "atencao"


# ---------- Classificação multiclasse ----------

def test_multiclasse_f1_alto_gera_bom():
    metrics = {"precisao": 0, "recall": 0, "f1_score": 0.85}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="multiclasse")
    assert resultado["status"] == "bom"


def test_multiclasse_f1_baixo_gera_ruim():
    metrics = {"precisao": 0, "recall": 0, "f1_score": 0.3}
    resultado = avaliar_modelo_classificacao(metrics, problem_type="multiclasse")
    assert resultado["status"] == "ruim"


# ---------- Regressão ----------

def test_r2_alto_gera_bom():
    resultado = avaliar_modelo_regressao({"r2": 0.95})
    assert resultado["status"] == "bom"


def test_r2_moderado_gera_atencao():
    resultado = avaliar_modelo_regressao({"r2": 0.65})
    assert resultado["status"] == "atencao"


def test_r2_baixo_gera_ruim():
    resultado = avaliar_modelo_regressao({"r2": 0.1})
    assert resultado["status"] == "ruim"


def test_r2_negativo_gera_ruim():
    resultado = avaliar_modelo_regressao({"r2": -0.5})
    assert resultado["status"] == "ruim"
    assert "pior" in resultado["titulo"].lower()
