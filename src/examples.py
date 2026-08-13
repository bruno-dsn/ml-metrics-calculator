from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARQUIVOS = {
    "Classificação binária": "exemplo_classificacao_binaria.csv",
    "Classificação multiclasse": "exemplo_classificacao_multiclasse.csv",
    "Regressão": "exemplo_regressao.csv",
}


def carregar_exemplo(tipo: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "sample_data" / ARQUIVOS[tipo])
