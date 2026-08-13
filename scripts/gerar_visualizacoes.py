from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import curva_limiares, matriz_confusao, metricas_binarias


FUNDO = "#090d18"
PAINEL = "#172033"
TEXTO = "#f8fafc"
MUTED = "#a8b5c8"
AZUL = "#38bdf8"
VERDE = "#34d399"
AMARELO = "#fbbf24"


def gerar() -> None:
    dados = pd.read_csv(ROOT / "sample_data" / "exemplo_classificacao_binaria.csv")
    metricas = metricas_binarias(dados["y_real"], dados["y_previsto"], 1, dados["y_proba"])
    matriz = matriz_confusao(dados["y_real"], dados["y_previsto"], [0, 1])
    limiares = curva_limiares(dados["y_real"], dados["y_proba"], 1)

    fig = plt.figure(figsize=(16, 9), facecolor=FUNDO)
    grade = fig.add_gridspec(12, 12, left=.055, right=.96, top=.76, bottom=.08, hspace=1.25, wspace=1.0)
    fig.text(.06, .955, "LABORATÓRIO DE MÉTRICAS", color="#7dd3fc", fontsize=10, fontweight="bold")
    fig.text(.06, .91, "Avalie o modelo além da acurácia", color=TEXTO, fontsize=25, fontweight="bold")
    fig.text(.06, .87, "Classe positiva explícita, limiar ajustável e leitura responsável dos erros.", color=MUTED, fontsize=11)

    cards = [
        ("ACURÁCIA BALANCEADA", metricas["acuracia_balanceada"]),
        ("PRECISÃO", metricas["precisao"]),
        ("RECALL", metricas["recall"]),
        ("F1-SCORE", metricas["f1"]),
    ]
    for i, (rotulo, valor) in enumerate(cards):
        ax = fig.add_subplot(grade[0:2, i * 3:(i + 1) * 3])
        ax.set_facecolor(PAINEL)
        ax.text(.06, .7, rotulo, transform=ax.transAxes, color=MUTED, fontsize=8, fontweight="bold")
        ax.text(.06, .2, f"{valor:.1%}", transform=ax.transAxes, color=TEXTO, fontsize=18, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#33445f")

    ax_matriz = fig.add_subplot(grade[3:10, :5])
    ax_matriz.set_facecolor(PAINEL)
    sns.heatmap(matriz, annot=True, fmt="d", cmap=sns.dark_palette(AZUL, as_cmap=True), cbar=False, ax=ax_matriz, annot_kws={"color": TEXTO, "fontsize": 13, "weight": "bold"})
    ax_matriz.set_title("Matriz de confusão", loc="left", color=TEXTO, fontsize=13, fontweight="bold", pad=12)
    ax_matriz.tick_params(colors=MUTED, labelsize=9)
    ax_matriz.set_xlabel("")
    ax_matriz.set_ylabel("")

    ax_limiar = fig.add_subplot(grade[3:10, 5:])
    ax_limiar.set_facecolor(PAINEL)
    ax_limiar.plot(limiares["limiar"], limiares["Precisão"], color=AZUL, linewidth=2.3, label="Precisão")
    ax_limiar.plot(limiares["limiar"], limiares["Recall"], color=VERDE, linewidth=2.3, label="Recall")
    ax_limiar.plot(limiares["limiar"], limiares["F1"], color=AMARELO, linewidth=2.3, label="F1")
    ax_limiar.axvline(.5, color="#fb7185", linestyle="--", linewidth=1.6, label="Limiar 0,50")
    ax_limiar.set_title("Impacto do limiar de decisão", loc="left", color=TEXTO, fontsize=13, fontweight="bold", pad=12)
    ax_limiar.set_xlabel("Limiar", color=MUTED, fontsize=9)
    ax_limiar.set_ylabel("Métrica", color=MUTED, fontsize=9)
    ax_limiar.set_ylim(0, 1.03)
    ax_limiar.grid(alpha=.17, color="#94a3b8")
    ax_limiar.tick_params(colors=MUTED, labelsize=8)
    ax_limiar.legend(frameon=False, labelcolor=TEXTO, ncol=2, fontsize=8)
    for spine in ax_limiar.spines.values():
        spine.set_color("#33445f")

    fig.text(.06, .052, "O resultado depende do custo dos erros e do contexto. O aplicativo não aplica um selo automático de qualidade.", color=MUTED, fontsize=8)
    (ROOT / "assets").mkdir(exist_ok=True)
    fig.savefig(ROOT / "assets" / "preview.png", dpi=150, facecolor=FUNDO)
    plt.close(fig)


if __name__ == "__main__":
    gerar()
    print("Visualização criada em assets/preview.png")
