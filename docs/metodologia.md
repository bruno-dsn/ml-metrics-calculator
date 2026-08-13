# Metodologia

## Classificação binária

A classe positiva é uma escolha explícita do usuário. Isso evita presumir que a classe rara ou o rótulo 1 representa necessariamente o evento mais importante.

Quando `y_proba` está disponível, o limiar converte probabilidade em classe prevista. Alterar esse valor muda falsos positivos, falsos negativos, precisão e recall.

O laboratório calcula:

- acurácia;
- acurácia balanceada;
- precisão;
- recall ou sensibilidade;
- especificidade;
- F1-score;
- ROC-AUC;
- PR-AUC.

## Classificação multiclasse

As médias macro calculam cada classe separadamente antes da média simples. Isso impede que uma classe grande domine completamente o resultado agregado. A aplicação também mostra métricas e suporte por classe.

## Regressão

- MAE mede o erro absoluto médio na unidade do alvo.
- RMSE aumenta a influência de erros grandes.
- R² compara o modelo com a previsão constante pela média.
- MAPE expressa erro relativo e exige cuidado com valores reais próximos de zero.

## Limites

Uma avaliação confiável também exige conhecer a estratégia de separação dos dados, o horizonte temporal, o custo dos erros, possíveis vieses e mudanças depois da implantação. Nenhuma métrica isolada define se um modelo está pronto para produção.
