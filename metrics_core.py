"""
metrics_core.py
================
Este módulo contém APENAS as contas (funções puras de cálculo).
Não tem nada de Streamlit, nem de gráfico, aqui, só matemática.

Por que separar assim?
-----------------------
1. Fica fácil escrever testes automáticos (arquivo tests/test_metrics.py),
   já que testamos números, não telas.
2. Se um dia você quiser trocar a interface (Streamlit -> Flask -> API),
   essas funções continuam funcionando sem mudar nada.

Cada função recebe:
    y_true -> os valores REAIS (o gabarito)
    y_pred -> os valores PREVISTOS pelo modelo

E devolve um dicionário Python com os resultados.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
    precision_recall_curve,
)


def classification_metrics(y_true, y_pred, problem_type: str = "binaria") -> dict:
    """
    Calcula acurácia, precisão, recall e F1-score.

    problem_type:
        "binaria"    -> duas classes (ex.: 0/1, "sim"/"nao")
        "multiclasse" -> três ou mais classes (ex.: "gato"/"cachorro"/"passaro")

    Para multiclasse, usamos a média "macro": calcula a métrica para
    cada classe separadamente e depois faz a média simples entre elas.
    Isso evita que uma classe muito grande "esconda" o desempenho ruim
    em uma classe pequena.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    average = "binary" if problem_type == "binaria" else "macro"

    # Em classificação binária, precision/recall/f1 precisam saber qual
    # rótulo é "a classe positiva". Se os rótulos já forem 0/1, o padrão
    # do scikit-learn (pos_label=1) funciona. Mas se forem strings como
    # "sim"/"nao" ou "fraude"/"legitima", isso quebrava antes; agora
    # assumimos que a classe positiva é a MENOS frequente em y_true
    # (convenção comum: o evento raro, ex. fraude, é o "positivo").
    pos_label_kwargs = {}
    if problem_type == "binaria":
        rotulos_unicos, contagens = np.unique(y_true, return_counts=True)
        if set(rotulos_unicos.tolist()) - {0, 1, "0", "1"}:
            pos_label_kwargs["pos_label"] = rotulos_unicos[np.argmin(contagens)]

    # zero_division=0 evita que o programa quebre quando uma classe
    # nunca é prevista (nesse caso a métrica daquela classe vira 0).
    resultado = {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, average=average, zero_division=0, **pos_label_kwargs),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0, **pos_label_kwargs),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0, **pos_label_kwargs),
    }
    return resultado


def calcular_matriz_confusao(y_true, y_pred):
    """Devolve a matriz de confusão e a lista de classes (labels), na mesma ordem."""
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    return matriz, labels


def regression_metrics(y_true, y_pred) -> dict:
    """
    Calcula as métricas clássicas de regressão:
        MAE  -> erro médio absoluto (fácil de interpretar: "em média, erramos X")
        RMSE -> parecido com o MAE, mas penaliza mais os erros grandes
        R²   -> de 0 a 1 (aproximadamente), indica o quanto o modelo explica
                a variação dos dados. 1 = perfeito, 0 = tão bom quanto
                simplesmente "chutar a média"
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def calcular_curva_roc(y_true, y_proba):
    """
    Calcula os pontos da curva ROC e a área sob a curva (AUC).
    Só faz sentido para classificação BINÁRIA, e precisa da probabilidade
    prevista (y_proba), não da classe final (0 ou 1).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    area = auc(fpr, tpr)
    return fpr, tpr, area


def calcular_curva_precision_recall(y_true, y_proba):
    """Calcula os pontos da curva Precision-Recall (também exige y_proba)."""
    precisao, recall, _ = precision_recall_curve(y_true, y_proba)
    return precisao, recall


def verificar_desbalanceamento(y_true, limite: float = 0.65) -> dict:
    """
    Verifica se as classes estão desbalanceadas.

    Regra simples: olhamos qual a proporção da classe mais frequente.
    Se ela representar mais que `limite` (padrão 65%) do total,
    consideramos o dataset desbalanceado.

    Retorna um dicionário com:
        desbalanceado (True/False)
        proporcoes: proporção de cada classe
        classe_majoritaria: qual classe é a mais comum
    """
    y_true = np.asarray(y_true)
    valores, contagens = np.unique(y_true, return_counts=True)
    proporcoes = {str(v): c / len(y_true) for v, c in zip(valores, contagens)}
    classe_majoritaria = str(valores[np.argmax(contagens)])
    proporcao_majoritaria = max(proporcoes.values())

    return {
        "desbalanceado": bool(proporcao_majoritaria >= limite),
        "proporcoes": proporcoes,
        "classe_majoritaria": classe_majoritaria,
        "proporcao_majoritaria": proporcao_majoritaria,
    }
