# Calculadora de Métricas de Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-informational?style=flat-square)

Ferramenta feita em Python e Streamlit que recebe os resultados de um
modelo (o que ele previu vs. o que era o valor real) e calcula as
métricas de avaliação, mostrando também **o que cada número
significa** e **quando ele pode te enganar**.

Serve para três tipos de problema:
- **Classificação binária** (2 classes, ex.: fraude / não-fraude)
- **Classificação multiclasse** (3+ classes, ex.: gato / cachorro / pássaro)
- **Regressão** (previsão de um número, ex.: preço de um imóvel)

Além das métricas em si, a aplicação traduz os números em um
**veredito automático**: uma caixa colorida (verde, amarela ou
vermelha) que resume se o modelo está bom, precisa de atenção, ou tem
desempenho fraco, incluindo para quais tipos de aplicação aquele
perfil de resultado é adequado ou arriscado de usar.

---

## Por que esse projeto existe

Muita gente (inclusive eu) decora a fórmula de precisão e recall, usa
uma vez no curso e esquece. A ideia aqui não é só "calcular a conta". É entregar uma ferramenta que **avisa quando a métrica está mentindo**.
Exemplo clássico: um dataset com 95% de exemplos da classe "não-fraude"
pode ter um modelo com 95% de acurácia que **nunca detecta uma fraude
de verdade**. A acurácia sozinha não mostra isso, o app avisa.

---

## Como rodar na sua máquina

```bash
# 1. Entre na pasta do projeto
cd ml-metrics-calculator

# 2. (Recomendado) crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode o app
streamlit run app.py
```

O Streamlit vai abrir automaticamente no navegador (normalmente em
`http://localhost:8501`). Se não abrir, copie o endereço que aparece
no terminal.

Não tem um CSV de resultados de modelo à mão? Marque a caixinha
**"Usar um CSV de exemplo"** na barra lateral: o app gera dados
fictícios na hora, só para você ver a ferramenta funcionando.

---

## Como preparar o seu CSV

### Classificação (binária ou multiclasse)

| y_real | y_previsto | y_proba (opcional, só binária) |
|--------|------------|----------------------------------|
| 1      | 1          | 0.91                              |
| 0      | 1          | 0.55                              |
| 1      | 0          | 0.32                              |

- `y_real`: o valor verdadeiro (o gabarito)
- `y_previsto`: a classe que o seu modelo previu
- `y_proba`: **opcional**, só faz sentido em classificação binária.
  É a probabilidade que o modelo deu para a classe positiva (o número
  entre 0 e 1 que normalmente sai de `model.predict_proba()` no
  scikit-learn). Com essa coluna, o app desenha a curva ROC e a curva
  Precision-Recall.

### Regressão

| y_real | y_previsto |
|--------|------------|
| 105000 | 98500      |
| 87000  | 91230      |

Tem exemplos prontos na pasta `sample_data/`, pode abrir um deles no
Excel/Google Sheets para entender o formato antes de gerar o seu.

---

## O que cada métrica quer dizer (resumo rápido)

**Classificação**
- **Acurácia**: % de acertos no total. Enganosa em dados desbalanceados.
- **Precisão**: das vezes que o modelo disse "positivo", quantas vezes
  estava certo. Precisão baixa = muitos falsos alarmes.
- **Recall**: dos casos positivos reais, quantos o modelo encontrou.
  Recall baixo = o modelo deixa passar casos importantes.
- **F1-score**: uma média entre precisão e recall, útil quando você
  quer um número só.

**Regressão**
- **MAE**: erro médio, em média o quanto a previsão erra (fácil de explicar).
- **RMSE**: parecido com o MAE, mas pune mais os erros grandes.
- **R²**: de 0 a 1 (pode ser negativo), diz o quanto o modelo explica
  a variação dos dados comparado a "só chutar a média".

O app explica cada uma dessas, calculada com os SEUS números, direto
na tela, vale mais a pena ler ali do que só aqui no README.

---

## Estrutura do projeto (e por que está organizado assim)

```
ml-metrics-calculator/
├── app.py              # A TELA (Streamlit). Só monta a interface.
├── metrics_core.py      # As CONTAS. Funções puras, sem interface.
├── explicacoes.py        # Os TEXTOS explicativos e alertas.
├── gerar_exemplos.py     # Script para gerar os CSVs de sample_data/
├── tests/
│   └── test_metrics.py   # Testes automáticos das funções de metrics_core.py
├── sample_data/          # CSVs de exemplo prontos para testar
└── requirements.txt
```

A separação entre "as contas" (`metrics_core.py`) e "a tela" (`app.py`)
é de propósito: assim é possível testar se a matemática está certa
sem precisar abrir o navegador, e é isso que o arquivo de testes faz.

---

## Rodando os testes automáticos

```bash
pip install -r requirements-dev.txt   # inclui o pytest
pytest -v
```

Isso confere, por exemplo, que quando o modelo acerta tudo a acurácia
dá 100%, que a matriz de confusão tem o formato certo, e que um
"modelo" que chuta um valor absurdo em regressão gera R² negativo.
Se algum dia você alterar `metrics_core.py` e quebrar alguma conta, os
testes avisam antes de você descobrir isso com um número errado na tela.

---

## Roadmap

- [x] Classificação binária (acurácia, precisão, recall, F1, matriz de confusão)
- [x] Curva ROC e Precision-Recall para classificação binária
- [x] Classificação multiclasse
- [x] Regressão (MAE, RMSE, R²)
- [x] Alerta automático de dataset desbalanceado
- [x] Veredito automático ("Model Assessment") com recomendação de uso
- [x] Testes automatizados (pytest)
- [ ] Exportação do relatório em PDF ou Markdown
- [ ] Aceitar múltiplos modelos no mesmo CSV para comparar lado a lado
- [ ] Métricas por classe individual na multiclasse (além da média macro)

## Link da aplicação

[Acesse a aplicação publicada no Streamlit Community Cloud](#) *(link a preencher após o deploy)*
