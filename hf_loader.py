from datasets import load_dataset
import pandas as pd


class HFDatasetLoader:

   # Clase para conectarse a Hugging Face Hub, obtener un dataset y devolverlo como un pandas DataFrame.
    def __init__(
        self,
        dataset_name: str,
        name: str | None = None,
        split: str | None = None,
    ):
        
        # :param dataset_name: Identificador del dataset en HF.
        # :param name: Subconjunto o tabla del dataset.
        # :param split: Split del dataset ("train", "test", "validation", ualquier otro mas).
        self.dataset_name = dataset_name
        self.name = name
        self.split = split
        self._dataset = None

    def load_data(self) -> pd.DataFrame:
        # Carga los datos desde Hugging Face y retorna un pandas DataFrame
        try:
            self._dataset = load_dataset(
                path=self.dataset_name, name=self.name, split=self.split
            )

            if isinstance(self._dataset, dict):
                first_split = list(self._dataset.keys())[0]
                df = self._dataset[first_split].to_pandas()
            else:
                df = self._dataset.to_pandas()

            return df

        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el dataset '{self.dataset_name}': {e}"
            )