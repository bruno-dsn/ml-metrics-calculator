from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def metricas_binarias(y_real, y_previsto, classe_positiva, y_proba=None) -> dict[str, float]:
    y_real_bin = np.asarray(y_real) == classe_positiva
    y_prev_bin = np.asarray(y_previsto) == classe_positiva
    matriz = confusion_matrix(y_real_bin, y_prev_bin, labels=[False, True])
    vn, fp, fn, vp = matriz.ravel()
    especificidade = vn / (vn + fp) if vn + fp else 0.0
    resultado = {
        "acuracia": accuracy_score(y_real_bin, y_prev_bin),
        "acuracia_balanceada": balanced_accuracy_score(y_real_bin, y_prev_bin),
        "precisao": precision_score(y_real_bin, y_prev_bin, zero_division=0),
        "recall": recall_score(y_real_bin, y_prev_bin, zero_division=0),
        "especificidade": especificidade,
        "f1": f1_score(y_real_bin, y_prev_bin, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_real_bin)) == 2:
        resultado["roc_auc"] = roc_auc_score(y_real_bin, y_proba)
        resultado["pr_auc"] = average_precision_score(y_real_bin, y_proba)
    return {chave: float(valor) for chave, valor in resultado.items()}


def metricas_multiclasse(y_real, y_previsto) -> tuple[dict[str, float], pd.DataFrame]:
    classes = sorted(set(y_real) | set(y_previsto), key=str)
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_real, y_previsto, labels=classes, zero_division=0
    )
    resumo = {
        "acuracia": float(accuracy_score(y_real, y_previsto)),
        "acuracia_balanceada": float(balanced_accuracy_score(y_real, y_previsto)),
        "precisao_macro": float(precision_score(y_real, y_previsto, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_real, y_previsto, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_real, y_previsto, average="macro", zero_division=0)),
    }
    por_classe = pd.DataFrame(
        {"classe": classes, "precisao": precisao, "recall": recall, "f1": f1, "suporte": suporte}
    )
    return resumo, por_classe


def metricas_regressao(y_real, y_previsto) -> dict[str, float]:
    real = np.asarray(y_real, dtype=float)
    previsto = np.asarray(y_previsto, dtype=float)
    mae = mean_absolute_error(real, previsto)
    rmse = np.sqrt(mean_squared_error(real, previsto))
    resultado = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2_score(real, previsto))}
    sem_zero = real != 0
    if sem_zero.any():
        resultado["mape"] = float(np.mean(np.abs((real[sem_zero] - previsto[sem_zero]) / real[sem_zero])))
    return resultado


def matriz_confusao(y_real, y_previsto, classes=None) -> pd.DataFrame:
    if classes is None:
        classes = sorted(set(y_real) | set(y_previsto), key=str)
    matriz = confusion_matrix(y_real, y_previsto, labels=classes)
    return pd.DataFrame(matriz, index=[f"Real: {c}" for c in classes], columns=[f"Previsto: {c}" for c in classes])


def aplicar_limiar(y_proba, limiar: float, classe_positiva, classe_negativa) -> np.ndarray:
    probabilidades = np.asarray(y_proba, dtype=float)
    if ((probabilidades < 0) | (probabilidades > 1)).any():
        raise ValueError("As probabilidades precisam estar entre 0 e 1.")
    return np.where(probabilidades >= limiar, classe_positiva, classe_negativa)


def curva_limiares(y_real, y_proba, classe_positiva) -> pd.DataFrame:
    real_binario = np.asarray(y_real) == classe_positiva
    linhas = []
    for limiar in np.linspace(0.05, 0.95, 19):
        previsto = np.asarray(y_proba) >= limiar
        linhas.append({
            "limiar": limiar,
            "Precisão": precision_score(real_binario, previsto, zero_division=0),
            "Recall": recall_score(real_binario, previsto, zero_division=0),
            "F1": f1_score(real_binario, previsto, zero_division=0),
        })
    return pd.DataFrame(linhas)


def dados_curva_roc(y_real, y_proba, classe_positiva) -> tuple[pd.DataFrame, float]:
    real_binario = np.asarray(y_real) == classe_positiva
    fpr, tpr, _ = roc_curve(real_binario, y_proba)
    return pd.DataFrame({"taxa_falso_positivo": fpr, "taxa_verdadeiro_positivo": tpr}), float(roc_auc_score(real_binario, y_proba))


def dados_curva_pr(y_real, y_proba, classe_positiva) -> tuple[pd.DataFrame, float]:
    real_binario = np.asarray(y_real) == classe_positiva
    precisao, recall, _ = precision_recall_curve(real_binario, y_proba)
    return pd.DataFrame({"recall": recall, "precisao": precisao}), float(average_precision_score(real_binario, y_proba))
