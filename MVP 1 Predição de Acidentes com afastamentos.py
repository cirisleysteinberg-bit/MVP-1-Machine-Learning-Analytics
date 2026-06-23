# MVP 1 PUC - Predição de Acidentes com Afastamento
# Autor: Cirisley Moraes
# Objetivo: treinar modelos de Machine Learning para prever Afastamento_Target (Sim/Não)

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

ARQUIVO = Path("base_ml_mvp1_acidentes.csv")
# Se estiver executando em outra pasta, altere para o caminho do arquivo CSV.

df = pd.read_csv(ARQUIVO, encoding="utf-8-sig")

# -----------------------------
# 1. Variável alvo
# -----------------------------
y = df["Afastamento_Target"].map({"Não": 0, "Sim": 1})

# -----------------------------
# 2. Variáveis de entrada
#    Atenção: remover colunas que vazam a resposta.
# -----------------------------
colunas_vazamento = [
    "Afastamento_Target",
    "Dias Perdidos",
    "Dias Debitados",
    "Classificação",
]
X = df.drop(columns=[c for c in colunas_vazamento if c in df.columns])

# Separar variáveis numéricas e categóricas
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# Pré-processamento
preprocessador = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

modelos = {
    "Regressao_Logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random_Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3,
    ),
}

resultados = []
matrizes = {}

for nome, modelo in modelos.items():
    pipe = Pipeline(steps=[
        ("preprocessador", preprocessador),
        ("modelo", modelo),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    resultados.append({
        "Modelo": nome,
        "Acuracia": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    })
    matrizes[nome] = confusion_matrix(y_test, y_pred)

    print("\n==============================")
    print(nome)
    print("==============================")
    print(classification_report(y_test, y_pred, target_names=["Não", "Sim"], zero_division=0))
    print("Matriz de Confusão:")
    print(matrizes[nome])

resultado_df = pd.DataFrame(resultados)
resultado_df.to_csv("resultados_modelos_mvp1.csv", index=False, encoding="utf-8-sig")
print("\nResultados salvos em: resultados_modelos_mvp1.csv")

# -----------------------------
# Importância das variáveis - Random Forest
# -----------------------------
rf_pipe = Pipeline(steps=[
    ("preprocessador", preprocessador),
    ("modelo", modelos["Random_Forest"]),
])
rf_pipe.fit(X_train, y_train)

feature_names = rf_pipe.named_steps["preprocessador"].get_feature_names_out()
importancias = rf_pipe.named_steps["modelo"].feature_importances_
imp = pd.DataFrame({"Variavel": feature_names, "Importancia": importancias})
imp = imp.sort_values("Importancia", ascending=False).head(20)
imp.to_csv("importancia_variaveis_mvp1.csv", index=False, encoding="utf-8-sig")
print("Importância das variáveis salva em: importancia_variaveis_mvp1.csv")
