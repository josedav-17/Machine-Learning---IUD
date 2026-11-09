from pathlib import Path
import pandas as pd


class DatasetSummarizer:
    
    
    # Clase para recibir un pandas DataFrame y generar un archivo .txt con un resumen detallado de su contenido real.
    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("El objeto recibido debe ser un pandas DataFrame.")

        self.df = df
        self.dataset_name = dataset_name

    def summarize_to_txt(
        self, output_folder: str = ".", filename: str | None = None
    ) -> str:
        # Genera el reporte .txt basado en las columnas y datos reales del DataFrame
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)

        base_name = (
            filename or f"{self.dataset_name.lower().replace(' ', '_')}_summary"
        )
        file_path = folder / f"{base_name}.txt"

        lines = []

        # Dimensiones generales
        lines.append("=" * 80)
        lines.append(
            f"REPORTE Y RESUMEN DEL DATASET: {self.dataset_name.upper()}"
        )
        lines.append("=" * 80)
        lines.append(f"Cantidad de Filas    : {len(self.df):,}")
        lines.append(f"Cantidad de Columnas : {self.df.shape[1]}")
        lines.append("-" * 80)

        # Estructura de columnas, tipos de datos y valores nulos
        lines.append("\n ESTRUCTURA DE LAS COLUMNAS Y VALORES NULOS:")
        lines.append("-" * 80)

        null_counts = self.df.isnull().sum()
        null_pct = (null_counts / len(self.df)) * 100 if len(self.df) > 0 else 0

        col_info = pd.DataFrame(
            {
                "Tipo de Dato": self.df.dtypes.astype(str),
                "Valores Nulos": null_counts,
                "% Nulos": null_pct.map("{:.2f}%".format),
            }
        )

        lines.append(col_info.to_string())
        lines.append("-" * 80)

        # 3Estadísticas Descriptivas Numéricas
        num_cols = self.df.select_dtypes(include=["number"])
        if not num_cols.empty:
            lines.append("\n ESTADÍSTICAS DESCRIPTIVAS (NUMÉRICAS):")
            lines.append("-" * 80)
            lines.append(num_cols.describe().T.to_string())
            lines.append("-" * 80)

        # Estadísticas Descriptivas Categóricas / Texto
        cat_cols = self.df.select_dtypes(
            include=["object", "category", "bool", "string"]
        )
        if not cat_cols.empty:
            lines.append("\n ESTADÍSTICAS DESCRIPTIVAS (CATEGÓRICAS / TEXTO):")
            lines.append("-" * 80)
            lines.append(cat_cols.describe().T.to_string())
            lines.append("-" * 80)

        # Vista previa de datos
        lines.append("\n ISTA PREVIA DE LOS DATOS (HEAD):")
        lines.append("-" * 80)
        lines.append(self.df.head().to_string())
        lines.append("\n" + "=" * 80)

        # Escribir el archivo
        content = "\n".join(lines)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Resumen generado: {file_path}")
        return str(file_path)


def write_execution_summary(resultados: dict) -> None:
    from datetime import datetime

    metricas = resultados["metricas"]
    importancias = resultados["importancias"]
    matriz_confusion = resultados["matriz_confusion"]
    modelo = resultados["modelo"]
    lines = [
        "RESUMEN AUTOMÁTICO DE LA EJECUCIÓN",
        "=" * 72,
        f"Fecha: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Pregunta de negocio: ¿Qué métricas cuantitativas durante un partido "
        "de fútbol (posesión, tiros a puerta y faltas) determinan con mayor "
        "probabilidad la victoria de un equipo?",
        "Beneficiario: cuerpo técnico, para orientar el análisis de partidos "
        "y la planificación de entrenamientos.",
        "Dataset: eatpizzanot/soccer-dataset, configuración match_stats; "
        "https://huggingface.co/datasets/eatpizzanot/soccer-dataset",
        "Fuente del resultado final: configuración fixtures, unida por "
        "fixture_id. Clases: victoria_local, empate y victoria_visitante.",
        "",
        "Tipo de aprendizaje: supervisado, clasificación de tres clases.",
        "Preprocesamiento: unión con resultados finales; se conservan partidos "
        "finalizados con goles disponibles; se excluyen identificadores, "
        "marcadores y goles al descanso de las variables predictoras; división "
        "estratificada 80/20 (semilla 42); imputación por mediana y "
        "estandarización ajustadas solo con X_train.",
        "Dimensiones antes de dividir: "
        f"({resultados['dimensiones_entrenamiento'][0] + resultados['dimensiones_prueba'][0]}, "
        f"{resultados['dimensiones_entrenamiento'][1]}).",
        f"Dimensiones de entrenamiento: {resultados['dimensiones_entrenamiento']}",
        f"Dimensiones de prueba: {resultados['dimensiones_prueba']}",
        f"Cantidad de variables predictoras: {len(resultados['variables'])}",
        "Variables predictoras: " + ", ".join(resultados["variables"]),
        "Distribución en entrenamiento: "
        + ", ".join(
            f"{clase}={cantidad}"
            for clase, cantidad in resultados["clases_entrenamiento"].items()
        ),
        "Distribución en prueba: "
        + ", ".join(
            f"{clase}={cantidad}"
            for clase, cantidad in resultados["clases_prueba"].items()
        ),
        "",
        "Modelo final: bosque aleatorio de clasificación.",
        f"Parámetros: n_estimators={modelo.n_estimators}, "
        f"max_depth={modelo.max_depth}, random_state={modelo.random_state}.",
        "Comparación de parámetros en una muestra estratificada de "
        "entrenamiento/validación (20.000/5.000 observaciones; F1 ponderada):",
    ]
    lines.extend(
        f"n_estimators={fila['n_estimators']}, max_depth={fila['max_depth']}: "
        f"{fila['F1 ponderada']:.4f}"
        for fila in resultados["comparacion"]
    )
    lines.extend(
        [
            "",
            "Medidas calculadas en el conjunto de prueba:",
        ]
    )
    lines.extend(f"{nombre}: {valor:.4f}" for nombre, valor in metricas.items())
    lines.extend(
        [
            "Referencia de exactitud al elegir siempre la clase más frecuente: "
            f"{resultados['referencia_clase_mayoritaria']:.4f}",
            "",
            "Matriz de confusión (filas: resultado real; "
            "columnas: resultado predicho):",
            matriz_confusion.to_string(),
            "",
            "Diez variables con mayor importancia:",
        ]
    )
    lines.extend(
        f"{nombre}: {valor:.4f}"
        for nombre, valor in importancias.head(10).items()
    )
    lines.append("=" * 72)

    outputs = Path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)
    contenido = "\n".join(lines) + "\n"
    (outputs / "resumen_ejecucion.txt").write_text(
        contenido, encoding="utf-8"
    )
    Path("resumen_ejecucion.txt").write_text(contenido, encoding="utf-8")