import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hf_loader import HFDatasetLoader


NOMBRES_EN_ESPANOL = {
    "fixture_id": "identificador del partido",
    "shots_total": "tiros totales",
    "shots_on_goal": "tiros a puerta",
    "shots_inside_box": "tiros dentro del área",
    "shots_outside_box": "tiros fuera del área",
    "blocked_shots": "tiros bloqueados",
    "penalties": "penaltis",
    "corners": "saques de esquina",
    "yellow_cards": "tarjetas amarillas",
    "red_cards": "tarjetas rojas",
    "xg": "goles esperados",
    "possession": "posesión",
    "fouls": "faltas",
    "offsides": "fueras de juego",
    "pass_accuracy": "precisión de pases",
    "goals_ht": "goles al descanso",
    "xg_ht": "goles esperados al descanso",
    "stats_fetched_at": "fecha de consulta",
    "in_csv": "incluido en CSV",
    "in_pq": "incluido en archivo Parquet",
    "xg_covered": "cobertura de goles esperados",
    "xg_nulled": "goles esperados sin datos",
    "known_at": "fecha de registro",
    "id": "identificador de la tabla de partidos",
    "goals_home": "goles locales",
    "goals_away": "goles visitantes",
    "status_norm": "estado del partido",
    "resultado": "resultado",
}


def traducir_columna(nombre):
    if nombre.startswith("home_") or nombre.startswith("away_"):
        lado, metrica = nombre.split("_", 1)
        nombre_metrica = NOMBRES_EN_ESPANOL.get(metrica, metrica)
        nombre_lado = "local" if lado == "home" else "visitante"
        return f"{nombre_metrica} {nombre_lado}"
    return NOMBRES_EN_ESPANOL.get(nombre, nombre)


def traducir_dataframe(df):
    return df.rename(columns=traducir_columna)


def preprocess(df: pd.DataFrame):
    estadisticas = [
        "shots_total",
        "shots_on_goal",
        "shots_inside_box",
        "shots_outside_box",
        "blocked_shots",
        "penalties",
        "corners",
        "yellow_cards",
        "red_cards",
        "xg",
        "possession",
        "fouls",
        "offsides",
        "pass_accuracy",
    ]
    columnas = [
        f"{equipo}_{estadistica}"
        for estadistica in estadisticas
        for equipo in ("home", "away")
    ]

    fixtures = HFDatasetLoader(
        "eatpizzanot/soccer-dataset", name="fixtures", split="train"
    ).load_data()
    df = df.merge(
        fixtures[["id", "goals_home", "goals_away", "status_norm"]],
        left_on="fixture_id",
        right_on="id",
        how="inner",
        validate="many_to_one",
    )
    df = df.loc[
        df["status_norm"].eq("FT")
        & df["goals_home"].notna()
        & df["goals_away"].notna()
    ].copy()

    df["resultado"] = "empate"
    df.loc[df["goals_home"] > df["goals_away"], "resultado"] = "victoria_local"
    df.loc[
        df["goals_home"] < df["goals_away"], "resultado"
    ] = "victoria_visitante"

    X = df[columnas]
    y = df["resultado"]
    print(f"Dimensiones antes de preparar los datos: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputador = SimpleImputer(strategy="median")
    X_train = imputador.fit_transform(X_train)
    X_test = imputador.transform(X_test)

    escalador = StandardScaler()
    X_train = escalador.fit_transform(X_train)
    X_test = escalador.transform(X_test)

    print(
        "Dimensiones después de preparar los datos: "
        f"X_train={X_train.shape}, X_test={X_test.shape}"
    )
    nombres_variables = [traducir_columna(nombre) for nombre in columnas]
    return X_train, X_test, y_train, y_test, nombres_variables, df
