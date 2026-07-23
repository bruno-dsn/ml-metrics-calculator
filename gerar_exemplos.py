"""
gerar_exemplos.py
===================
Gera os 3 CSVs de exemplo em sample_data/, para quem baixar o projeto
poder testar a ferramenta sem precisar ter um modelo treinado.

Rode com:
    python3 gerar_exemplos.py
"""

import numpy as np
import pandas as pd


def gerar_exemplo(problem_type: str, variante: str = "bom") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if problem_type == "binaria":
        n = 200
        y_real = rng.choice([0, 1], size=n, p=[0.85, 0.15])
        if variante == "ruim":
            # proba quase independente do rótulo real -> modelo ruim, quase aleatório
            y_proba = np.clip(rng.normal(0.5, 0.25, n), 0, 1)
        else:
            y_proba = np.clip(y_real * 0.6 + rng.normal(0, 0.25, n) + 0.15, 0, 1)
        y_previsto = (y_proba >= 0.5).astype(int)
        return pd.DataFrame({"y_real": y_real, "y_previsto": y_previsto, "y_proba": y_proba})
    elif problem_type == "multiclasse":
        n = 150
        classes = ["gato", "cachorro", "passaro"]
        y_real = rng.choice(classes, size=n, p=[0.4, 0.4, 0.2])
        y_previsto = y_real.copy()
        erro_idx = rng.choice(n, size=int(n * 0.25), replace=False)
        y_previsto[erro_idx] = rng.choice(classes, size=len(erro_idx))
        return pd.DataFrame({"y_real": y_real, "y_previsto": y_previsto})
    else:
        n = 100
        y_real = rng.normal(50, 15, n)
        y_previsto = y_real + rng.normal(0, 5, n)
        return pd.DataFrame({"y_real": y_real, "y_previsto": y_previsto})


if __name__ == "__main__":
    mapa = {
        ("binaria", "bom"): "classificacao_binaria",
        ("binaria", "ruim"): "classificacao_binaria_modelo_ruim",
        ("multiclasse", "bom"): "classificacao_multiclasse",
        ("regressao", "bom"): "regressao",
    }
    for (tipo, variante), nome in mapa.items():
        df = gerar_exemplo(tipo, variante)
        caminho = f"sample_data/exemplo_{nome}.csv"
        df.to_csv(caminho, index=False)
        print(f"Gerado {caminho} com {len(df)} linhas")
