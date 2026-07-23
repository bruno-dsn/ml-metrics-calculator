"""
explicacoes.py
===============
Aqui moram só os TEXTOS explicativos. A ideia é: cada função recebe os
números já calculados (vindos de metrics_core.py) e devolve uma lista
de strings explicando o que aquilo significa, e alertando quando um
número pode enganar quem está olhando.

Separar os textos da lógica de cálculo deixa mais fácil ajustar a
linguagem depois, sem tocar nas contas.
"""


def explicar_classificacao(metrics: dict, desbalanceamento: dict, problem_type: str) -> list[str]:
    textos = []

    acc = metrics["acuracia"]
    textos.append(
        f"**Acurácia = {acc:.2%}** → de tudo que o modelo previu, essa é a "
        f"fração de acertos (certos e errados, somados)."
    )

    if desbalanceamento["desbalanceado"]:
        classe = desbalanceamento["classe_majoritaria"]
        prop = desbalanceamento["proporcao_majoritaria"]
        textos.append(
            f"⚠️ **Atenção: dataset desbalanceado.** A classe **{classe}** "
            f"representa {prop:.0%} dos dados. Nesse cenário, a acurácia pode "
            f"parecer ótima mesmo com um modelo ruim, basta ele sempre "
            f"'chutar' a classe majoritária. Olhe com mais cuidado o "
            f"**recall** e a **precisão** de cada classe, e considere o F1-score "
            f"como referência principal."
        )

    precisao = metrics["precisao"]
    textos.append(
        f"**Precisão = {precisao:.2%}** → das vezes que o modelo disse "
        f"'positivo', essa fração realmente era positiva. Precisão baixa "
        f"= muitos falsos positivos (o modelo 'grita fogo' sem necessidade)."
    )

    recall = metrics["recall"]
    textos.append(
        f"**Recall = {recall:.2%}** → dos casos que realmente eram "
        f"positivos, essa fração o modelo conseguiu encontrar. Recall baixo "
        f"= muitos falsos negativos (o modelo deixa passar casos "
        f"importantes (perigoso em diagnóstico médico ou fraude, por exemplo)."
    )

    f1 = metrics["f1_score"]
    textos.append(
        f"**F1-score = {f1:.2%}** → uma média entre precisão e recall. É "
        f"útil quando você quer um único número que não deixe nenhuma das "
        f"duas métricas cair sem chamar atenção. Especialmente relevante "
        f"em dados desbalanceados."
    )

    if precisao > 0.85 and recall < 0.5:
        textos.append(
            "**Padrão notado:** precisão alta + recall baixo. O modelo é "
            "'conservador': só aponta positivo quando tem bastante certeza, "
            "mas deixa passar muitos casos positivos reais."
        )
    elif recall > 0.85 and precisao < 0.5:
        textos.append(
            "**Padrão notado:** recall alto + precisão baixa. O modelo é "
            "'alarmista': encontra quase todos os positivos reais, mas "
            "aponta positivo demais, gerando muitos falsos alarmes."
        )

    if problem_type == "multiclasse":
        textos.append(
            "ℹ️ Como este é um problema **multiclasse**, precisão, recall e "
            "F1 foram calculados por classe e depois tirada a média simples "
            "entre elas ('macro average'), para que uma classe grande não "
            "esconda o desempenho ruim em uma classe pequena."
        )

    return textos


def explicar_regressao(metrics: dict, amplitude_y: float | None = None) -> list[str]:
    textos = []

    mae = metrics["mae"]
    textos.append(
        f"**MAE (Erro Médio Absoluto) = {mae:.4f}** → em média, a previsão "
        f"do modelo erra por essa quantidade, para cima ou para baixo. É a "
        f"métrica mais fácil de explicar para alguém fora da área de dados."
    )

    rmse = metrics["rmse"]
    textos.append(
        f"**RMSE (Raiz do Erro Quadrático Médio) = {rmse:.4f}** → parecido "
        f"com o MAE, mas eleva os erros ao quadrado antes de tirar a média, "
        f"o que faz erros grandes pesarem muito mais. Se RMSE >> MAE, é sinal "
        f"de que existem alguns erros bem grandes (outliers) puxando o RMSE "
        f"para cima."
    )

    r2 = metrics["r2"]
    textos.append(
        f"**R² = {r2:.4f}** → indica o quanto o modelo explica a variação "
        f"dos dados, indo (na prática) de 0 a 1. R² = 1 seria previsão "
        f"perfeita; R² = 0 significa que o modelo não é melhor do que "
        f"simplesmente prever a média de tudo. **R² negativo é possível**: "
        f"significa que o modelo é PIOR do que só chutar a média, sinal de "
        f"que algo está errado no treinamento."
    )

    if amplitude_y is not None and amplitude_y > 0:
        erro_relativo = mae / amplitude_y
        if erro_relativo > 0.2:
            textos.append(
                f"⚠️ O MAE representa {erro_relativo:.0%} da amplitude dos "
                f"valores reais (diferença entre o maior e o menor valor). "
                f"Isso é um erro relativamente grande, vale investigar se "
                f"faltam variáveis importantes no modelo."
            )

    return textos


# ---------------------------------------------------------------------
# Veredito automático ("Model Assessment")
# ---------------------------------------------------------------------
# A ideia aqui é diferente das funções acima: em vez de explicar cada
# métrica separadamente, essas funções traduzem os números em UMA
# conclusão prática, incluindo para quais tipos de aplicação o modelo
# atual é adequado ou arriscado de usar.
#
# Os limiares (0.5, 0.8 etc.) são regras práticas, não valores
# absolutos. Servem para dar uma leitura rápida, não substituem uma
# análise mais aprofundada do problema de negócio.

def avaliar_modelo_classificacao(metrics: dict, problem_type: str = "binaria") -> dict:
    """
    Traduz precisão, recall e F1 em um veredito prático.

    Retorna um dicionário com:
        status         -> "bom", "atencao" ou "ruim" (usado para escolher a cor da caixa)
        titulo         -> resumo de uma linha
        resumo         -> explicação de 1 a 2 frases
        adequado_para  -> lista de cenários onde esse perfil de modelo funciona bem
        evitar_para    -> lista de cenários onde esse perfil de modelo é arriscado
    """
    precisao = metrics["precisao"]
    recall = metrics["recall"]
    f1 = metrics["f1_score"]

    if problem_type == "multiclasse":
        if f1 >= 0.8:
            return {
                "status": "bom",
                "titulo": "Desempenho bom",
                "resumo": f"F1-score macro de {f1:.0%}, o modelo está consistente entre as classes.",
                "adequado_para": [],
                "evitar_para": [],
            }
        if f1 >= 0.5:
            return {
                "status": "atencao",
                "titulo": "Desempenho moderado",
                "resumo": (
                    f"F1-score macro de {f1:.0%}. Vale olhar o desempenho por "
                    f"classe individualmente, a média pode esconder uma classe fraca."
                ),
                "adequado_para": [],
                "evitar_para": [],
            }
        return {
            "status": "ruim",
            "titulo": "Desempenho fraco",
            "resumo": f"F1-score macro de {f1:.0%} é baixo, o modelo provavelmente precisa de mais dados ou ajuste.",
            "adequado_para": [],
            "evitar_para": [],
        }

    # Classificação binária
    if precisao >= 0.8 and recall < 0.5:
        return {
            "status": "atencao",
            "titulo": "Alta precisão, recall baixo",
            "resumo": "O modelo raramente erra quando aponta positivo, mas deixa passar muitos casos positivos reais.",
            "adequado_para": ["Filtro de spam", "Recomendações, onde um erro custa pouco"],
            "evitar_para": ["Diagnóstico médico", "Detecção de fraude", "Qualquer caso onde deixar passar um positivo é grave"],
        }
    if recall >= 0.8 and precisao < 0.5:
        return {
            "status": "atencao",
            "titulo": "Recall alto, precisão baixa",
            "resumo": "O modelo encontra quase todos os casos positivos reais, mas gera muitos falsos alarmes.",
            "adequado_para": ["Triagem inicial com revisão humana depois", "Casos onde não pode faltar um positivo"],
            "evitar_para": ["Decisão automática sem revisão humana", "Casos onde um falso alarme tem custo alto"],
        }
    if precisao >= 0.8 and recall >= 0.8:
        return {
            "status": "bom",
            "titulo": "Modelo equilibrado e adequado",
            "resumo": "Boa precisão e bom recall ao mesmo tempo, o modelo acerta a maioria dos positivos sem gerar muitos falsos alarmes.",
            "adequado_para": ["A maioria dos casos de uso de classificação binária"],
            "evitar_para": [],
        }
    if precisao < 0.5 and recall < 0.5:
        return {
            "status": "ruim",
            "titulo": "Desempenho fraco",
            "resumo": "Precisão e recall baixos ao mesmo tempo, o modelo provavelmente precisa de mais dados, features novas ou ajuste de hiperparâmetros.",
            "adequado_para": [],
            "evitar_para": ["Uso em produção no estado atual"],
        }
    return {
        "status": "atencao",
        "titulo": "Desempenho moderado",
        "resumo": "Nem ótimo nem ruim. Vale comparar com um modelo baseline simples antes de decidir se compensa usar.",
        "adequado_para": [],
        "evitar_para": [],
    }


def avaliar_modelo_regressao(metrics: dict) -> dict:
    """Traduz o R² em um veredito prático para modelos de regressão."""
    r2 = metrics["r2"]

    if r2 >= 0.8:
        return {
            "status": "bom",
            "titulo": "Bom ajuste aos dados",
            "resumo": f"R² de {r2:.2f}, o modelo explica bem a variação dos dados.",
            "adequado_para": [],
            "evitar_para": [],
        }
    if r2 >= 0.5:
        return {
            "status": "atencao",
            "titulo": "Ajuste moderado",
            "resumo": f"R² de {r2:.2f}. O modelo captura parte do padrão, mas ainda há bastante erro não explicado.",
            "adequado_para": [],
            "evitar_para": [],
        }
    if r2 >= 0:
        return {
            "status": "ruim",
            "titulo": "Ajuste fraco",
            "resumo": f"R² de {r2:.2f} está próximo de zero, o modelo melhora pouco em relação a simplesmente prever a média.",
            "adequado_para": [],
            "evitar_para": [],
        }
    return {
        "status": "ruim",
        "titulo": "Modelo pior do que a média",
        "resumo": f"R² negativo ({r2:.2f}) significa que prever a média dos dados seria melhor do que este modelo. Vale revisar o treinamento.",
        "adequado_para": [],
        "evitar_para": [],
    }
