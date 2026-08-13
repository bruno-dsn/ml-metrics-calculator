from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    grafico_limiares,
    grafico_matriz,
    grafico_metricas_classe,
    grafico_pr,
    grafico_regressao,
    grafico_residuos,
    grafico_roc,
)
from src.examples import carregar_exemplo
from src.metrics import (
    aplicar_limiar,
    curva_limiares,
    dados_curva_pr,
    dados_curva_roc,
    matriz_confusao,
    metricas_binarias,
    metricas_multiclasse,
    metricas_regressao,
)
from src.validation import diagnosticar_distribuicao, inferir_classes, validar_colunas


st.set_page_config(page_title="Laboratório de Métricas de ML", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #090d18 0%, #111827 42%, #101827 100%);}
    [data-testid="stSidebar"] {background: #0d1320; border-right: 1px solid #28354a;}
    .block-container {max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.6rem 1.8rem; border: 1px solid #2d4161; border-radius: 18px; background: linear-gradient(135deg, rgba(37,99,235,.25), rgba(124,58,237,.19)); margin-bottom: 1.25rem;}
    .hero-kicker {color: #7dd3fc; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;}
    .hero h1 {color: #f8fafc; font-size: 2.25rem; margin: .25rem 0 .35rem;}
    .hero p {color: #cbd5e1; margin: 0; max-width: 900px;}
    [data-testid="stMetric"] {background: rgba(23,32,51,.88); border: 1px solid #33445f; padding: .95rem; border-radius: 14px;}
    [data-testid="stMetricLabel"] {color: #a8b5c8;}
    [data-testid="stMetricValue"] {color: #f8fafc;}
    div[data-baseweb="tab-list"] {gap: .5rem;}
    button[data-baseweb="tab"] {background: #172033; border-radius: 10px 10px 0 0; padding: .7rem 1rem;}
    .note {border-left: 4px solid #38bdf8; padding: .85rem 1rem; background: rgba(56,189,248,.08); border-radius: 0 10px 10px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Configuração")
tipo = st.sidebar.radio("Tipo de problema", ["Classificação binária", "Classificação multiclasse", "Regressão"])
origem = st.sidebar.radio("Entrada", ["Exemplo do projeto", "Enviar CSV"])

if origem == "Exemplo do projeto":
    dados_brutos = carregar_exemplo(tipo)
    st.sidebar.caption("Dados demonstrativos identificados, disponíveis em sample_data.")
else:
    arquivo = st.sidebar.file_uploader("Arquivo CSV", type=["csv"])
    if arquivo is None:
        st.info("Envie um CSV para começar ou selecione o exemplo do projeto.")
        st.stop()
    dados_brutos = pd.read_csv(arquivo)

try:
    dados = validar_colunas(dados_brutos, tipo)
except ValueError as erro:
    st.error(str(erro))
    st.stop()

st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Avaliação responsável de modelos</div>
      <h1>Laboratório de métricas de Machine Learning</h1>
      <p>Analise desempenho, escolha a classe positiva e observe como o limiar altera erros, precisão e recall.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.expander("Conferir dados de entrada"):
    st.dataframe(dados.head(20), use_container_width=True)
    st.caption(f"{len(dados):,} registros válidos e {len(dados.columns)} colunas.")


def mostrar_percentuais(valores: list[tuple[str, float]]) -> None:
    colunas = st.columns(len(valores))
    for coluna, (nome, valor) in zip(colunas, valores):
        coluna.metric(nome, f"{valor:.1%}")


if tipo == "Classificação binária":
    classes = inferir_classes(dados)
    if len(classes) != 2:
        st.error(f"Foram encontradas {len(classes)} classes. Selecione classificação multiclasse ou revise os dados.")
        st.stop()
    classe_positiva = st.sidebar.selectbox("Classe positiva", classes, index=1)
    classe_negativa = next(classe for classe in classes if classe != classe_positiva)
    tem_proba = "y_proba" in dados.columns and dados["y_proba"].notna().all()
    limiar = st.sidebar.slider("Limiar de decisão", 0.05, 0.95, 0.50, 0.05, disabled=not tem_proba)
    if tem_proba:
        st.sidebar.caption(f"y_proba será interpretada como a probabilidade da classe {classe_positiva}.")
        previsto = aplicar_limiar(dados["y_proba"], limiar, classe_positiva, classe_negativa)
    else:
        previsto = dados["y_previsto"]
        st.sidebar.caption("Inclua y_proba para testar outros limiares e visualizar ROC e Precision-Recall.")

    metricas = metricas_binarias(dados["y_real"], previsto, classe_positiva, dados["y_proba"] if tem_proba else None)
    mostrar_percentuais([
        ("Acurácia balanceada", metricas["acuracia_balanceada"]),
        ("Precisão", metricas["precisao"]),
        ("Recall", metricas["recall"]),
        ("F1-score", metricas["f1"]),
    ])

    diagnostico = diagnosticar_distribuicao(dados["y_real"])
    if diagnostico["desbalanceado"]:
        st.warning(f"A classe majoritária representa {diagnostico['proporcao_majoritaria']:.1%} dos registros. Por isso, a acurácia balanceada merece mais atenção que a acurácia simples.")

    aba_erros, aba_limiar, aba_curvas, aba_metodo = st.tabs(["Erros", "Limiar", "Curvas", "Como interpretar"])
    with aba_erros:
        esquerda, direita = st.columns([1, 1])
        with esquerda:
            st.subheader("Matriz de confusão")
            matriz = matriz_confusao(dados["y_real"], previsto, classes)
            st.altair_chart(grafico_matriz(matriz), use_container_width=True)
        with direita:
            st.subheader("Leitura das métricas")
            st.markdown(f"""
            <div class="note">
            <strong>Precisão:</strong> entre as previsões da classe {classe_positiva}, quantas estavam corretas.<br><br>
            <strong>Recall:</strong> entre os casos reais da classe {classe_positiva}, quantos foram encontrados.<br><br>
            <strong>Especificidade:</strong> entre os casos da classe {classe_negativa}, quantos foram corretamente descartados.<br><br>
            A importância de falso positivo e falso negativo depende do problema de negócio.
            </div>
            """, unsafe_allow_html=True)
    with aba_limiar:
        if tem_proba:
            st.altair_chart(grafico_limiares(curva_limiares(dados["y_real"], dados["y_proba"], classe_positiva), limiar), use_container_width=True)
            st.caption("A linha vertical mostra o limiar selecionado. Não existe limiar universalmente melhor.")
        else:
            st.info("Esta análise exige a coluna y_proba.")
    with aba_curvas:
        if tem_proba:
            roc, auc_roc = dados_curva_roc(dados["y_real"], dados["y_proba"], classe_positiva)
            pr, auc_pr = dados_curva_pr(dados["y_real"], dados["y_proba"], classe_positiva)
            c1, c2 = st.columns(2)
            c1.altair_chart(grafico_roc(roc, auc_roc), use_container_width=True)
            c2.altair_chart(grafico_pr(pr, auc_pr), use_container_width=True)
            st.caption("Em eventos raros, a curva Precision-Recall costuma ser mais informativa que a ROC.")
        else:
            st.info("As curvas exigem probabilidades na coluna y_proba.")
    with aba_metodo:
        st.write("A aplicação não atribui um selo automático de modelo bom ou ruim. O resultado depende do custo de cada erro, da distribuição dos dados, do período de validação e do uso pretendido.")
        st.code("y_real, y_previsto, y_proba", language="text")

elif tipo == "Classificação multiclasse":
    resumo, por_classe = metricas_multiclasse(dados["y_real"], dados["y_previsto"])
    mostrar_percentuais([
        ("Acurácia", resumo["acuracia"]),
        ("Acurácia balanceada", resumo["acuracia_balanceada"]),
        ("Precisão macro", resumo["precisao_macro"]),
        ("F1 macro", resumo["f1_macro"]),
    ])
    aba_classe, aba_matriz, aba_metodo = st.tabs(["Métricas por classe", "Matriz de confusão", "Como interpretar"])
    with aba_classe:
        st.altair_chart(grafico_metricas_classe(por_classe), use_container_width=True)
        st.dataframe(por_classe.style.format({"precisao": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"}), use_container_width=True)
    with aba_matriz:
        st.altair_chart(grafico_matriz(matriz_confusao(dados["y_real"], dados["y_previsto"])), use_container_width=True)
    with aba_metodo:
        st.write("A média macro calcula cada classe separadamente e depois obtém a média simples. Assim, uma classe numerosa não esconde completamente o desempenho das classes menores.")

else:
    metricas = metricas_regressao(dados["y_real"], dados["y_previsto"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metricas['mae']:.3f}")
    c2.metric("RMSE", f"{metricas['rmse']:.3f}")
    c3.metric("R²", f"{metricas['r2']:.3f}")
    c4.metric("MAPE", f"{metricas.get('mape', float('nan')):.1%}" if "mape" in metricas else "Indisponível")
    aba_ajuste, aba_residuos, aba_metodo = st.tabs(["Real versus previsto", "Resíduos", "Como interpretar"])
    with aba_ajuste:
        st.altair_chart(grafico_regressao(dados), use_container_width=True)
        st.caption("Quanto mais próximos os pontos estiverem da linha diagonal, menor é o erro.")
    with aba_residuos:
        st.altair_chart(grafico_residuos(dados), use_container_width=True)
        st.caption("Padrões nos resíduos podem indicar que o modelo não capturou parte da estrutura dos dados.")
    with aba_metodo:
        st.markdown("""
        - **MAE:** erro absoluto médio, na mesma unidade do alvo.
        - **RMSE:** penaliza erros grandes com mais intensidade.
        - **R²:** compara o modelo com a previsão pela média. Pode ser negativo.
        - **MAPE:** erro percentual médio. Deve ser evitado quando o valor real é zero ou muito próximo de zero.
        """)

st.divider()
st.caption("Projeto educacional. Métricas descrevem dados de avaliação e não substituem validação temporal, análise de viés ou monitoramento em produção.")
