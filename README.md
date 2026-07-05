[README_MVP_Machine_Learning_Analytics.md](https://github.com/user-attachments/files/29675602/README_MVP_Machine_Learning_Analytics.md)
# MVP — Machine Learning & Analytics

## Predição de Acidentes com Afastamento no Sistema Elétrico de Potência (SEP)

**Autor:** Cirisley Ferreira de Moraes  
**Matrícula:** 4052026000477  
**Data:** Julho/2026  
**Tipo de problema:** Classificação binária  
**Dataset:** Base histórica de acidentes e indicadores de segurança do trabalho no SEP  

---

## 1. Objetivo do projeto

Este MVP tem como objetivo construir, avaliar e comparar modelos de Machine Learning capazes de prever se um acidente de trabalho no Sistema Elétrico de Potência (SEP) resultará ou não em **afastamento do colaborador**.

A proposta utiliza dados históricos de acidentes e características operacionais dos eventos, como empresa, segmento, tipo de trabalho, local, agente causador, potencial, grau de risco, horário e outras variáveis relacionadas.

O modelo deve apoiar uma abordagem mais preventiva de Segurança e Saúde no Trabalho (SST), auxiliando na identificação de cenários com maior probabilidade de acidentes com afastamento.

---

## 2. Problema de negócio

As atividades de operação e manutenção no setor elétrico envolvem riscos críticos, como eletricidade, trabalho em altura, espaços confinados, equipamentos de grande porte e sistemas energizados.

Além disso, o alto turnover da mão de obra, especialmente em empresas contratadas, pode aumentar a exposição ao risco, pois muitos trabalhadores chegam às atividades com baixa experiência prática no SEP e menor percepção dos perigos.

Atualmente, grande parte da gestão de acidentes ocorre de forma reativa, após a ocorrência do evento. Este MVP busca demonstrar como dados históricos podem ser utilizados para apoiar decisões preventivas.

---

## 3. Tipo de problema de Machine Learning

O problema é de **classificação binária supervisionada**.

A variável alvo é:

| Classe | Significado |
|---|---|
| `Sim` | O acidente gerou afastamento |
| `Não` | O acidente não gerou afastamento |

Não se trata de regressão, pois o objetivo não é prever um valor numérico contínuo. Também não é clusterização, pois existe uma variável alvo conhecida.

---

## 4. Hipóteses do MVP

As hipóteses principais são:

1. As características registradas no momento do acidente possuem relação com a chance de afastamento.
2. Modelos de Machine Learning conseguem identificar padrões nos dados históricos.
3. Um modelo candidato, como Random Forest, pode ter desempenho superior a um baseline simples.
4. As variáveis mais importantes podem apoiar a priorização de ações preventivas em SST.

---

## 5. Critérios de sucesso

A métrica principal escolhida é o **F1-Score da classe `Sim`**, pois o problema pode apresentar desbalanceamento entre acidentes com e sem afastamento.

Também devem ser analisadas as métricas:

- Accuracy;
- Precision;
- Recall;
- F1-Score;
- Matriz de confusão.

No contexto de segurança do trabalho, o **Recall da classe `Sim`** é especialmente importante, pois falsos negativos representam acidentes com afastamento previstos incorretamente como sem afastamento.

---

## 6. Estrutura sugerida do repositório

```text
MVP-1-Machine-Learning-Analytics/
│
├── README.md
├── MVP_Machine_Learning_Analytics.ipynb
├── base_ml_mvp1_acidentes.csv
│
├── artefatos_mvp1/
│   ├── pipeline_random_forest_otimizado.pkl
│   ├── resultados_modelos.csv
│   ├── resultados_gridsearch.csv
│   ├── importancia_variaveis.csv
│   ├── matriz_confusao_modelo_final.png
│   └── importancia_variaveis.png
│
└── requirements.txt
```

---

## 7. Dataset

O arquivo esperado é:

```text
base_ml_mvp1_acidentes.csv
```

No notebook, a base deve ser carregada por URL pública do GitHub, por exemplo:

```python
DATA_URL = "https://raw.githubusercontent.com/cirisleysteinberg-bit/MVP-1-Machine-Learning-Analytics/main/base_ml_mvp1_acidentes.csv"
```

Antes da entrega, confirme que o arquivo CSV está publicado no repositório e que a URL está funcionando.

---

## 8. Principais variáveis do projeto

As principais variáveis utilizadas no modelo incluem:

- Empresa;
- Segmento;
- Tipo de Trabalho;
- Local;
- Município;
- Estado;
- Instalação;
- Diretoria;
- Gerência Executiva;
- Gerência;
- Dia da Semana;
- Hora;
- Hora_Inicial;
- Agente Causador;
- Tipo de Lesão;
- Parte do Corpo Atingida;
- Gravidade;
- Potencial;
- Grau de Risco;
- Compromissos (Regra de Ouro).

As colunas abaixo **não devem ser usadas como entrada do modelo**, pois podem gerar vazamento de dados:

- Dias Perdidos;
- Dias Debitados;
- Afastamento;
- Afastamento_Target;
- Empregado;
- Ano;
- Mês;
- Dia.

---

## 9. Etapas que devem ser executadas no MVP

### 9.1 Definir o problema

Descrever o contexto do SEP, os riscos das atividades, o problema dos acidentes de trabalho e o objetivo de prever acidentes com afastamento.

### 9.2 Carregar os dados

Carregar o CSV a partir da URL pública do GitHub ou, em testes locais, a partir do arquivo no mesmo diretório do notebook.

### 9.3 Entender os dados

Executar:

```python
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()
df.shape
```

Verificar:

- quantidade de linhas e colunas;
- tipos das variáveis;
- valores ausentes;
- registros duplicados;
- consistência da variável alvo.

### 9.4 Criar ou validar a variável alvo

A variável alvo deve indicar se houve afastamento.

Exemplo:

```python
df["Dias Perdidos"] = pd.to_numeric(df["Dias Perdidos"], errors="coerce").fillna(0)
df["Afastamento"] = np.where(df["Dias Perdidos"] > 0, "Sim", "Não")
```

### 9.5 Fazer análise exploratória

Gerar gráficos e tabelas para analisar:

- distribuição da variável alvo;
- frequência por empresa, segmento e tipo de trabalho;
- frequência por agente causador;
- relação entre potencial, grau de risco e afastamento;
- possíveis categorias com maior proporção de afastamento.

### 9.6 Preparar os dados

Selecionar as variáveis explicativas e remover colunas com risco de vazamento.

Separar:

```python
X = df[features]
y = df["Afastamento"]
```

### 9.7 Dividir treino e teste

Usar holdout com estratificação:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### 9.8 Criar pipeline de pré-processamento

O pipeline deve tratar:

- variáveis numéricas com imputação pela mediana e padronização;
- variáveis categóricas com imputação pela moda e One-Hot Encoding.

### 9.9 Treinar modelos

Treinar e comparar pelo menos três abordagens:

1. DummyClassifier — baseline mínimo;
2. Regressão Logística — baseline estatístico;
3. Random Forest — modelo candidato.

### 9.10 Avaliar modelos

Calcular:

- Accuracy;
- Precision;
- Recall;
- F1-Score;
- Matriz de confusão;
- Classification Report.

### 9.11 Otimizar hiperparâmetros

Usar GridSearchCV no Random Forest com validação cruzada estratificada.

A métrica de otimização deve ser o F1-Score da classe `Sim`.

### 9.12 Comparar resultados

Criar uma tabela final comparando os modelos e indicar qual apresentou melhor equilíbrio entre F1-Score, Recall e generalização.

### 9.13 Analisar variáveis importantes

Para o Random Forest final, calcular e visualizar as variáveis mais importantes.

### 9.14 Salvar artefatos

Salvar os arquivos finais na pasta `artefatos_mvp1`.

Artefatos esperados:

```text
pipeline_random_forest_otimizado.pkl
resultados_modelos.csv
resultados_gridsearch.csv
importancia_variaveis.csv
matriz_confusao_modelo_final.png
importancia_variaveis.png
```

---

## 10. Como executar no Google Colab

1. Abrir o notebook no Google Colab.
2. Confirmar que o arquivo CSV está disponível no GitHub.
3. Atualizar a variável `DATA_URL` com o link raw do CSV.
4. Executar as células na ordem.
5. Conferir se o dataset foi carregado corretamente.
6. Verificar se os gráficos foram gerados.
7. Conferir a tabela de comparação dos modelos.
8. Verificar se os artefatos foram salvos na pasta `artefatos_mvp1`.
9. Validar se o notebook roda do início ao fim sem erro.

---

## 11. Bibliotecas utilizadas

O projeto utiliza bibliotecas comuns do Google Colab:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Opcionalmente, pode ser criado um arquivo `requirements.txt` com:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

## 12. Modelos utilizados

### DummyClassifier

Modelo de referência mínima. Serve para verificar se os modelos realmente aprendem padrões além da classe majoritária.

### Regressão Logística

Modelo simples, interpretável e adequado como baseline estatístico para classificação binária.

### Random Forest

Modelo candidato, capaz de capturar relações não lineares e interações entre variáveis. Também permite análise de importância das variáveis.

---

## 13. Métrica principal

A métrica principal é:

```text
F1-Score da classe Sim
```

Justificativa:

- equilibra Precision e Recall;
- é mais adequada quando existe desbalanceamento entre classes;
- valoriza o desempenho na classe de maior interesse: acidentes com afastamento.

---

## 14. Cuidados importantes

Durante o desenvolvimento, observar os seguintes cuidados:

- Não usar `Dias Perdidos` como variável explicativa, pois ela define o afastamento;
- Não usar `Dias Debitados`, pois representa consequência do acidente;
- Evitar vazamento de dados;
- Usar pipelines para garantir reprodutibilidade;
- Usar `random_state=42`;
- Manter o notebook executável do início ao fim;
- Explicar as decisões técnicas em texto;
- Interpretar os resultados com visão crítica;
- Não afirmar que o modelo substitui especialistas de SST.

---

## 15. Limitações do MVP

As principais limitações são:

- base histórica de uma única organização;
- possível desbalanceamento da variável alvo;
- dados sujeitos a inconsistências de preenchimento;
- ausência de variáveis ambientais e comportamentais;
- possibilidade de categorias pouco frequentes;
- generalização limitada para outros cenários.

O modelo deve ser tratado como ferramenta de apoio à decisão, e não como decisão automática.

---

## 16. Próximos passos recomendados

Para evoluir o projeto, recomenda-se:

- ampliar a base histórica;
- incluir indicadores de exposição ao risco, como HHT;
- testar modelos como XGBoost, LightGBM e CatBoost;
- aplicar técnicas de interpretabilidade, como SHAP e LIME;
- criar dashboard em Power BI;
- monitorar o desempenho do modelo ao longo do tempo;
- validar os resultados com especialistas de Segurança do Trabalho.

---

## 17. Checklist final de entrega

Antes de entregar o MVP, confirmar:

- [ ] O problema está claramente descrito;
- [ ] O tipo de problema foi definido como classificação binária;
- [ ] O dataset está disponível por URL pública;
- [ ] A variável alvo foi criada ou validada;
- [ ] A análise exploratória foi realizada;
- [ ] A divisão treino/teste foi feita com estratificação;
- [ ] Variáveis com vazamento foram removidas;
- [ ] O pré-processamento foi feito com pipeline;
- [ ] Existe pelo menos um baseline;
- [ ] Foram comparados modelos diferentes;
- [ ] Foi feita otimização de hiperparâmetros;
- [ ] As métricas foram calculadas corretamente;
- [ ] A matriz de confusão foi apresentada;
- [ ] As limitações foram discutidas;
- [ ] Os artefatos foram salvos;
- [ ] O notebook executa do início ao fim;
- [ ] A conclusão está conectada ao objetivo inicial.

---

## 18. Conclusão

Este MVP demonstra como técnicas de Machine Learning podem ser aplicadas a dados históricos de acidentes do SEP para prever a ocorrência de afastamento.

A solução proposta contribui para uma gestão mais preventiva, apoiando a priorização de ações, inspeções, treinamentos e revisões de procedimentos. Apesar disso, o modelo deve ser utilizado como apoio à decisão, sempre complementado pela análise técnica de profissionais de Segurança do Trabalho, Operação e Manutenção.
