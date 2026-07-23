"""
app.py
=======
Interface Streamlit da Calculadora de Métricas de Machine Learning.

Como rodar:
    streamlit run app.py

O que este arquivo faz (e o que ele NÃO faz):
- Ele só monta a TELA: lê o que o usuário escolheu, chama as funções de
  metrics_core.py e explicacoes.py, e desenha o resultado.
- Ele não tem nenhuma fórmula de métrica dentro dele. Se quiser entender
  as contas, olhe metrics_core.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from metrics_core import (
    classification_metrics,
    calcular_matriz_confusao,
    regression_metrics,
    calcular_curva_roc,
    calcular_curva_precision_recall,
    verificar_desbalanceamento,
)
from explicacoes import (
    explicar_classificacao,
    explicar_regressao,
    avaliar_modelo_classificacao,
    avaliar_modelo_regressao,
)
from gerar_exemplos import gerar_exemplo

st.set_page_config(page_title="Calculadora de Métricas de ML", page_icon="📊", layout="wide")

st.title("Calculadora de Métricas de Machine Learning")
st.caption(
    "Suba um CSV com as previsões do seu modelo e receba as métricas certas, "
    "com gráficos e explicação de quando cada uma pode te enganar."
)

# -----------------------------------------------------------------
# 1. Escolha do tipo de problema
# -----------------------------------------------------------------
st.sidebar.header("1. Tipo de problema")
problem_type = st.sidebar.radio(
    "O que o seu modelo faz?",
    options=["binaria", "multiclasse", "regressao"],
    format_func=lambda x: {
        "binaria": "Classificação binária (2 classes)",
        "multiclasse": "Classificação multiclasse (3+ classes)",
        "regressao": "Regressão (número contínuo)",
    }[x],
)

st.sidebar.header("2. Envie o CSV")

if problem_type in ("binaria", "multiclasse"):
    st.sidebar.markdown(
        "O CSV precisa ter as colunas:\n"
        "- `y_real` → a classe verdadeira\n"
        "- `y_previsto` → a classe que o modelo previu\n\n"
        "Para classificação **binária**, você pode incluir também a coluna "
        "opcional `y_proba` (probabilidade prevista da classe positiva) "
        "para desbloquear a curva ROC e a curva Precision-Recall."
    )
else:
    st.sidebar.markdown(
        "O CSV precisa ter as colunas:\n"
        "- `y_real` → o valor numérico verdadeiro\n"
        "- `y_previsto` → o valor numérico previsto pelo modelo"
    )

arquivo = st.sidebar.file_uploader("Arquivo CSV", type=["csv"])

usar_exemplo = st.sidebar.checkbox("Usar um CSV de exemplo em vez de enviar um arquivo")

st.sidebar.header("3. Ou baixe um CSV de exemplo")
if problem_type == "binaria":
    st.sidebar.download_button(
        "📥 Baixar exemplo (modelo bom)",
        data=gerar_exemplo("binaria", "bom").to_csv(index=False),
        file_name="classificacao_binaria.csv",
        mime="text/csv",
    )
    st.sidebar.download_button(
        "📥 Baixar exemplo (modelo ruim)",
        data=gerar_exemplo("binaria", "ruim").to_csv(index=False),
        file_name="classificacao_binaria_modelo_ruim.csv",
        mime="text/csv",
        help="Útil para comparar com o resultado do modelo bom.",
    )
elif problem_type == "multiclasse":
    st.sidebar.download_button(
        "📥 Baixar CSV de exemplo",
        data=gerar_exemplo("multiclasse").to_csv(index=False),
        file_name="classificacao_multiclasse.csv",
        mime="text/csv",
    )
else:
    st.sidebar.download_button(
        "📥 Baixar CSV de exemplo",
        data=gerar_exemplo("regressao").to_csv(index=False),
        file_name="regressao.csv",
        mime="text/csv",
    )


df = None
if usar_exemplo:
    df = gerar_exemplo(problem_type)
    st.info("Usando um CSV de exemplo gerado automaticamente (dados fictícios).")
elif arquivo is not None:
    try:
        df = pd.read_csv(arquivo)
    except Exception as e:
        st.error(
            f"Não consegui ler esse arquivo como CSV. Verifique se é um "
            f"arquivo `.csv` válido (separado por vírgula). Detalhe técnico: {e}"
        )
        st.stop()

if df is None:
    st.warning("Envie um CSV na barra lateral, ou marque a opção de usar um exemplo, para começar.")
    st.stop()


def exibir_veredito(veredito: dict) -> None:
    """Isso serve para: mostrar o veredito automático numa caixa colorida
    (verde = bom, amarelo = atenção, vermelho = ruim), usando os
    componentes nativos do Streamlit em vez de HTML customizado."""
    caixa_por_status = {"bom": st.success, "atencao": st.warning, "ruim": st.error}
    caixa = caixa_por_status[veredito["status"]]

    texto = f"**{veredito['titulo']}**  \n{veredito['resumo']}"
    if veredito["adequado_para"]:
        texto += "\n\n**Adequado para:** " + "; ".join(veredito["adequado_para"]) + "."
    if veredito["evitar_para"]:
        texto += "\n\n**Evitar para:** " + "; ".join(veredito["evitar_para"]) + "."

    caixa(texto)

colunas_necessarias = {"y_real", "y_previsto"}
if not colunas_necessarias.issubset(df.columns):
    st.error(f"O CSV precisa ter as colunas {colunas_necessarias}. Encontrado: {list(df.columns)}")
    st.stop()

linhas_antes = len(df)
df = df.dropna(subset=list(colunas_necessarias))
if len(df) < linhas_antes:
    st.warning(
        f"{linhas_antes - len(df)} linha(s) com valores vazios em `y_real` ou "
        f"`y_previsto` foram ignoradas."
    )
if df.empty:
    st.error("O CSV não tem nenhuma linha válida para calcular métricas.")
    st.stop()

if problem_type == "regressao":
    for coluna in ("y_real", "y_previsto"):
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            st.error(
                f"Para regressão, a coluna `{coluna}` precisa conter só números. "
                f"Encontrei valores não numéricos nela."
            )
            st.stop()

with st.expander("Ver as primeiras linhas do CSV"):
    st.dataframe(df.head(10))

y_real = df["y_real"]
y_previsto = df["y_previsto"]

# -----------------------------------------------------------------
# 2. Classificação (binária ou multiclasse)
# -----------------------------------------------------------------
if problem_type in ("binaria", "multiclasse"):
    metrics = classification_metrics(y_real, y_previsto, problem_type=problem_type)
    desbalanceamento = verificar_desbalanceamento(y_real)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{metrics['acuracia']:.2%}")
    col2.metric("Precisão", f"{metrics['precisao']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-score", f"{metrics['f1_score']:.2%}")

    st.subheader("Avaliação geral")
    veredito = avaliar_modelo_classificacao(metrics, problem_type=problem_type)
    exibir_veredito(veredito)

    st.subheader("O que esses números significam")
    for texto in explicar_classificacao(metrics, desbalanceamento, problem_type):
        st.markdown(texto)

    st.subheader("Matriz de confusão")
    matriz, labels = calcular_matriz_confusao(y_real, y_previsto)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Previsto pelo modelo")
    ax.set_ylabel("Valor real")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                     color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Nº de exemplos")
    st.pyplot(fig)
    if desbalanceamento["desbalanceado"]:
        st.caption(
            f"⚠️ Os blocos desta matriz têm tamanhos bem diferentes porque as "
            f"classes do seu conjunto de dados são desbalanceadas: a classe "
            f"'{desbalanceamento['classe_majoritaria']}' representa "
            f"{desbalanceamento['proporcao_majoritaria']:.0%} dos exemplos. "
            f"Isso não é um erro do modelo nem da matriz — é reflexo direto "
            f"da quantidade de exemplos que cada classe tem no seu CSV."
        )
    st.caption(
        "Linhas = classe verdadeira, colunas = classe prevista. A diagonal "
        "principal (canto superior-esquerdo até o inferior-direito) mostra "
        "os acertos; fora dela estão os erros do modelo."
    )

    # ROC e Precision-Recall só para binário + com y_proba
    if problem_type == "binaria" and "y_proba" in df.columns:
        st.subheader("Curva ROC e Curva Precision-Recall")
        col_a, col_b = st.columns(2)

        fpr, tpr, area = calcular_curva_roc(y_real, df["y_proba"])
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {area:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Modelo aleatório")
        ax_roc.set_xlabel("Falso Positivo (FPR)")
        ax_roc.set_ylabel("Verdadeiro Positivo (TPR / Recall)")
        ax_roc.set_title("Curva ROC")
        ax_roc.legend()
        col_a.pyplot(fig_roc)
        col_a.caption(
            "Quanto mais a curva 'abraça' o canto superior esquerdo, melhor. "
            "A linha pontilhada é o desempenho de um modelo que chuta aleatoriamente."
        )

        precisao_c, recall_c = calcular_curva_precision_recall(y_real, df["y_proba"])
        fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
        ax_pr.plot(recall_c, precisao_c)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precisão")
        ax_pr.set_title("Curva Precision-Recall")
        col_b.pyplot(fig_pr)
        col_b.caption(
            "Especialmente útil em dados desbalanceados, mostra o "
            "'trade-off' entre encontrar mais positivos (recall) e "
            "acertar quando aponta positivo (precisão)."
        )
    elif problem_type == "binaria":
        st.info(
            "💡 Inclua uma coluna `y_proba` (probabilidade prevista) no CSV "
            "para também ver a curva ROC e a curva Precision-Recall."
        )

# -----------------------------------------------------------------
# 3. Regressão
# -----------------------------------------------------------------
else:
    metrics = regression_metrics(y_real, y_previsto)
    amplitude = float(np.max(y_real) - np.min(y_real))

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['mae']:.4f}")
    col2.metric("RMSE", f"{metrics['rmse']:.4f}")
    col3.metric("R²", f"{metrics['r2']:.4f}")

    st.subheader("Avaliação geral")
    veredito = avaliar_modelo_regressao(metrics)
    exibir_veredito(veredito)

    st.subheader("O que esses números significam")
    for texto in explicar_regressao(metrics, amplitude_y=amplitude):
        st.markdown(texto)

    st.subheader("Real vs. Previsto")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_real, y_previsto, alpha=0.6)
    limite_min = min(y_real.min(), y_previsto.min())
    limite_max = max(y_real.max(), y_previsto.max())
    ax.plot([limite_min, limite_max], [limite_min, limite_max], linestyle="--", color="gray",
            label="Previsão perfeita")
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor previsto")
    ax.legend()
    st.pyplot(fig)
    st.caption(
        "Quanto mais os pontos ficam colados na linha pontilhada (previsão "
        "perfeita), melhor o modelo. Pontos muito acima ou abaixo da linha "
        "são os maiores erros."
    )

st.divider()
st.caption(
    "Projeto educacional feito para praticar métricas de avaliação de "
    "modelos de Machine Learning. Código aberto, veja o README no "
    "repositório para entender como cada métrica é calculada."
)