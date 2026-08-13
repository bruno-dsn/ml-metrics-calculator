from __future__ import annotations

import pandas as pd


def validar_colunas(dados: pd.DataFrame, tipo: str) -> pd.DataFrame:
    obrigatorias = {"y_real", "y_previsto"}
    ausentes = obrigatorias - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes: {', '.join(sorted(ausentes))}.")
    resultado = dados.dropna(subset=list(obrigatorias)).copy()
    if resultado.empty:
        raise ValueError("Não há linhas válidas depois da remoção de valores ausentes.")
    if tipo == "Regressão":
        for coluna in obrigatorias:
            resultado[coluna] = pd.to_numeric(resultado[coluna], errors="coerce")
        resultado = resultado.dropna(subset=list(obrigatorias))
        if len(resultado) < 2:
            raise ValueError("A regressão precisa de pelo menos duas observações numéricas.")
    if "y_proba" in resultado.columns:
        resultado["y_proba"] = pd.to_numeric(resultado["y_proba"], errors="coerce")
    return resultado


def inferir_classes(dados: pd.DataFrame) -> list:
    return sorted(set(dados["y_real"]) | set(dados["y_previsto"]), key=str)


def diagnosticar_distribuicao(y_real) -> dict:
    contagens = pd.Series(y_real).value_counts(dropna=False)
    proporcoes = contagens / contagens.sum()
    return {
        "quantidade_classes": int(len(contagens)),
        "classe_majoritaria": contagens.index[0],
        "proporcao_majoritaria": float(proporcoes.iloc[0]),
        "desbalanceado": bool(proporcoes.iloc[0] >= 0.70),
    }
