[![Python checks](https://github.com/bruno-dsn/ml-metrics-calculator/actions/workflows/tests.yml/badge.svg)](https://github.com/bruno-dsn/ml-metrics-calculator/actions/workflows/tests.yml)

# Laboratório de Métricas de Machine Learning

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-aplicação-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-métricas-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-gráficos-1F77B4?style=for-the-badge)
![Tests](https://img.shields.io/badge/testes-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/licença-MIT-0F766E?style=for-the-badge)

[Acessar aplicação](https://ml-metrics-calculator.streamlit.app/)

Aplicação educacional para interpretar métricas de classificação e regressão com contexto. O laboratório permite escolher a classe positiva, ajustar o limiar de decisão e observar como cada escolha altera os erros do modelo.

![Visão do laboratório](assets/preview.png)

## Problema

Acurácia alta não garante um modelo útil. Em uma base com evento raro, um classificador pode acertar a maioria e ainda ignorar quase todos os casos importantes.

O projeto ajuda a responder:

1. Qual classe está sendo tratada como positiva?
2. Quantos positivos reais o modelo encontrou?
3. Quantos alertas positivos estavam corretos?
4. Como o resultado muda quando o limiar é alterado?
5. O desempenho está concentrado em uma classe?
6. Os erros de regressão apresentam algum padrão?

## Funcionalidades

### Classificação binária

- escolha explícita da classe positiva;
- limiar ajustável quando `y_proba` está presente;
- acurácia balanceada, precisão, recall, especificidade e F1;
- matriz de confusão;
- ROC-AUC e PR-AUC;
- curva de precisão, recall e F1 por limiar;
- alerta de distribuição desbalanceada.

### Classificação multiclasse

- acurácia e acurácia balanceada;
- médias macro;
- precisão, recall, F1 e suporte por classe;
- matriz de confusão completa.

### Regressão

- MAE, RMSE, R² e MAPE;
- gráfico de valor real versus previsto;
- análise visual dos resíduos.

## Mudança metodológica importante

A aplicação não atribui mais um selo automático de modelo bom ou ruim. Limites fixos ignoram o custo de cada erro e o contexto de uso. A nova versão apresenta evidências e explica como interpretá-las.

## Formato do CSV

Classificação:

```csv
y_real,y_previsto,y_proba
0,0,0.08
1,1,0.91
1,0,0.42
```

`y_proba` é opcional e representa a probabilidade da classe positiva escolhida.

Regressão:

```csv
y_real,y_previsto
105000,98500
87000,91230
```

## Estrutura

```text
ml-metrics-calculator/
├── app.py
├── assets/
│   └── preview.png
├── docs/
│   ├── como_explicar_o_projeto.md
│   ├── linkedin.md
│   └── metodologia.md
├── sample_data/
├── scripts/
│   └── gerar_visualizacoes.py
├── src/
│   ├── charts.py
│   ├── examples.py
│   ├── metrics.py
│   └── validation.py
└── tests/
```

## Executar com Python 3.14

```bash
git clone https://github.com/bruno-dsn/ml-metrics-calculator.git
cd ml-metrics-calculator

python3.14 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

No Windows, ative o ambiente com `.venv\Scripts\activate`.

## Testes

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Os testes cobrem métricas, classe positiva textual, limiar, probabilidades, multiclasse, regressão, validação dos dados e abertura do aplicativo.

## Limitações

O laboratório recebe resultados já produzidos por um modelo. Ele não verifica vazamento de dados, estratégia de validação, estabilidade temporal, viés, latência ou monitoramento em produção.

## Autor

**Bruno Nunes**

Estudante da Pós-Tech AI Scientist na FIAP, com foco em Ciência de Dados, Machine Learning e produtos orientados a dados.

[GitHub](https://github.com/bruno-dsn) | [LinkedIn](https://www.linkedin.com/in/bruno-dsnunes/)

## Licença

Distribuído sob a licença MIT.


## Verificação automatizada

O workflow [Python checks](.github/workflows/tests.yml) instala as dependências de desenvolvimento e executa a suíte de testes em Python 3.12 a cada push ou pull request. O badge acima mostra o resultado real da execução, sem um número fixo de testes.
