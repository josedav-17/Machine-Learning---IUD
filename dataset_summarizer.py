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