from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def train_model(X_train, X_test, y_train, y_test, feature_names):
    X_ajuste, X_validacion, y_ajuste, y_validacion = train_test_split(
        X_train,
        y_train,
        train_size=20000,
        test_size=5000,
        random_state=42,
        stratify=y_train,
    )
    configuraciones = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 50, "max_depth": 15},
        {"n_estimators": 100, "max_depth": 10},
    ]
    comparacion = []
    for parametros in configuraciones:
        modelo_prueba = RandomForestClassifier(
            **parametros, random_state=42
        )
        modelo_prueba.fit(X_ajuste, y_ajuste)
        prediccion_validacion = modelo_prueba.predict(X_validacion)
        comparacion.append(
            {
                **parametros,
                "F1 ponderada": f1_score(
                    y_validacion,
                    prediccion_validacion,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "Exactitud": accuracy_score(y_test, y_pred),
        "Precisión ponderada": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Sensibilidad ponderada": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Medida F1 ponderada": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    print("\nResultados de la evaluación:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")

    outputs = Path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    matriz_confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=modelo.classes_),
        index=modelo.classes_,
        columns=modelo.classes_,
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz_confusion.to_numpy(),
        display_labels=[
            nombre.replace("_", " ").capitalize()
            for nombre in modelo.classes_
        ],
    )
    display.plot(cmap="Blues", values_format="d", xticks_rotation=30)
    plt.title("Matriz de resultados reales y predichos")
    plt.tight_layout()
    plt.savefig(outputs / "confusion_matrix.png", dpi=150)
    plt.close()

    importancias = pd.Series(
        modelo.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    principales = importancias.head(15).sort_values()
    plt.figure(figsize=(9, 6))
    principales.plot(kind="barh")
    plt.xlabel("Nivel de importancia")
    plt.title("15 variables con mayor importancia")
    plt.tight_layout()
    plt.savefig(outputs / "feature_importance.png", dpi=150)
    plt.close()

    print(f"\nGráficas guardadas en: {outputs.resolve()}")
    return {
        "modelo": modelo,
        "metricas": metricas,
        "importancias": importancias,
        "matriz_confusion": matriz_confusion,
        "comparacion": comparacion,
        "variables": feature_names,
        "clases_entrenamiento": y_train.value_counts().to_dict(),
        "clases_prueba": y_test.value_counts().to_dict(),
        "dimensiones_entrenamiento": X_train.shape,
        "dimensiones_prueba": X_test.shape,
        "referencia_clase_mayoritaria": y_test.value_counts(normalize=True).max(),
    }
