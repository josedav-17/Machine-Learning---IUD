from dataset_summarizer import write_execution_summary
from hf_loader import HFDatasetLoader
from preprocess import preprocess, traducir_dataframe
from train_model import train_model

loader = HFDatasetLoader(
    dataset_name="eatpizzanot/soccer-dataset",
    name="match_stats",
    split="train",
)
df = loader.load_data()
print(f"Dimensiones del conjunto de datos: {df.shape}")
df_espanol = traducir_dataframe(df)
print(f"Columnas: {df_espanol.columns.tolist()}")
tipo_por_clave = {
    "i": "entero",
    "u": "entero",
    "f": "decimal",
    "b": "lógico",
    "M": "fecha y hora",
    "O": "texto u otros",
}
tipos = df_espanol.dtypes.map(
    lambda tipo: tipo_por_clave.get(tipo.kind, "texto u otros")
)
tipos_texto = "\n".join(
    f"{nombre}: {tipo}" for nombre, tipo in tipos.items()
)
print(f"Tipos de datos:\n{tipos_texto}")
print(f"Primeros 5 registros:\n{df_espanol.head()}")

X_train, X_test, y_train, y_test, nombres_variables, df = preprocess(df)
resultados = train_model(X_train, X_test, y_train, y_test, nombres_variables)
write_execution_summary(resultados)
