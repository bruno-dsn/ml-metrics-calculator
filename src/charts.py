from __future__ import annotations

import altair as alt
import pandas as pd


AZUL = "#38bdf8"
VERDE = "#34d399"
AMARELO = "#fbbf24"
ROSA = "#fb7185"


def grafico_matriz(matriz: pd.DataFrame) -> alt.Chart:
    dados = matriz.rename_axis("real").reset_index().melt("real", var_name="previsto", value_name="quantidade")
    base = alt.Chart(dados).encode(x=alt.X("previsto:N", title=None), y=alt.Y("real:N", title=None))
    mapa = base.mark_rect().encode(
        color=alt.Color("quantidade:Q", scale=alt.Scale(range=["#172033", AZUL]), title="Registros"),
        tooltip=["real:N", "previsto:N", "quantidade:Q"],
    )
    texto = base.mark_text(fontSize=14, fontWeight="bold").encode(text="quantidade:Q", color=alt.value("#f8fafc"))
    return (mapa + texto).properties(height=340)


def grafico_limiares(dados: pd.DataFrame, limiar_atual: float) -> alt.Chart:
    longo = dados.melt("limiar", var_name="métrica", value_name="valor")
    linhas = alt.Chart(longo).mark_line(strokeWidth=2.4).encode(
        x=alt.X("limiar:Q", title="Limiar de decisão", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("valor:Q", title="Métrica", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("métrica:N", scale=alt.Scale(range=[AZUL, VERDE, AMARELO])),
        tooltip=[alt.Tooltip("limiar:Q", format=".2f"), "métrica:N", alt.Tooltip("valor:Q", format=".2%")],
    )
    regra = alt.Chart(pd.DataFrame({"limiar": [limiar_atual]})).mark_rule(color=ROSA, strokeDash=[5, 4], strokeWidth=2).encode(x="limiar:Q")
    return (linhas + regra).properties(height=350)


def grafico_roc(dados: pd.DataFrame, auc: float) -> alt.Chart:
    linha = alt.Chart(dados).mark_line(color=AZUL, strokeWidth=2.4).encode(
        x=alt.X("taxa_falso_positivo:Q", title="Taxa de falsos positivos", axis=alt.Axis(format=".0%")),
        y=alt.Y("taxa_verdadeiro_positivo:Q", title="Taxa de verdadeiros positivos", axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("taxa_falso_positivo:Q", format=".2%"), alt.Tooltip("taxa_verdadeiro_positivo:Q", format=".2%")],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#64748b", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    return (linha + diagonal).properties(height=330, title=f"Curva ROC, área = {auc:.3f}")


def grafico_pr(dados: pd.DataFrame, area: float) -> alt.Chart:
    return alt.Chart(dados).mark_line(color=VERDE, strokeWidth=2.4).encode(
        x=alt.X("recall:Q", title="Recall", axis=alt.Axis(format=".0%")),
        y=alt.Y("precisao:Q", title="Precisão", axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("recall:Q", format=".2%"), alt.Tooltip("precisao:Q", format=".2%")],
    ).properties(height=330, title=f"Curva Precision-Recall, área = {area:.3f}")


def grafico_metricas_classe(dados: pd.DataFrame) -> alt.Chart:
    longo = dados.melt(["classe", "suporte"], value_vars=["precisao", "recall", "f1"], var_name="métrica", value_name="valor")
    return alt.Chart(longo).mark_bar().encode(
        x=alt.X("classe:N", title="Classe"),
        y=alt.Y("valor:Q", title="Métrica", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("métrica:N", scale=alt.Scale(range=[AZUL, VERDE, AMARELO])),
        xOffset="métrica:N",
        tooltip=["classe:N", "métrica:N", alt.Tooltip("valor:Q", format=".2%"), "suporte:Q"],
    ).properties(height=360)


def grafico_regressao(dados: pd.DataFrame) -> alt.Chart:
    minimo = float(min(dados["y_real"].min(), dados["y_previsto"].min()))
    maximo = float(max(dados["y_real"].max(), dados["y_previsto"].max()))
    pontos = alt.Chart(dados).mark_circle(size=80, color=AZUL, opacity=.72).encode(
        x=alt.X("y_real:Q", title="Valor real", scale=alt.Scale(zero=False)),
        y=alt.Y("y_previsto:Q", title="Valor previsto", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("y_real:Q", format=".3f"), alt.Tooltip("y_previsto:Q", format=".3f")],
    )
    ideal = alt.Chart(pd.DataFrame({"x": [minimo, maximo], "y": [minimo, maximo]})).mark_line(color=ROSA, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    return (pontos + ideal).properties(height=380)


def grafico_residuos(dados: pd.DataFrame) -> alt.Chart:
    tabela = dados.copy()
    tabela["residuo"] = tabela["y_real"] - tabela["y_previsto"]
    pontos = alt.Chart(tabela).mark_circle(size=70, color=AMARELO, opacity=.72).encode(
        x=alt.X("y_previsto:Q", title="Valor previsto", scale=alt.Scale(zero=False)),
        y=alt.Y("residuo:Q", title="Resíduo"),
        tooltip=[alt.Tooltip("y_previsto:Q", format=".3f"), alt.Tooltip("residuo:Q", format=".3f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=ROSA, strokeDash=[5, 4]).encode(y="y:Q")
    return (pontos + zero).properties(height=380)
