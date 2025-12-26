from typing import Literal
from pathlib import Path

import pandas as pd

from src.database.db_connect import DataBaseProjectSQLAlchemy


def write_data_to_db(config_file: dict, input_data_path: Path) -> None:
    """
    Escritura de tablas en la database. Se escriben las tablas iniciales del proyecto
    """

    # Instanciamos la clase de DB
    db = DataBaseProjectSQLAlchemy(db_name=config_file["database"]["database_name"])
    db.db_connect()  # conecta la database (si no está creada, se crea en el momento)

    dataframes = {}
    for table_name, table in config_file["input"]["tables"].items():
        if table_name != "default_info":
            table_path = input_data_path / table  # añade directorio donde está el input
            df = pd.read_csv(table_path)
            dataframes[table_name] = df
            db.write_table(df, config_file["database"]["tables_name"][table_name], "replace")
        else:
            for table_id, real_table in table.items():
                table_path = input_data_path / real_table
                df = pd.read_csv(table_path)
                dataframes[table_id] = df
                db.write_table(df, config_file["database"]["tables_name"][table_name][table_id], "replace")

    # check dataframes
    print(dataframes.keys())
    for df in dataframes.values():
        isinstance(df, pd.DataFrame)


def write_model_data_to_db(
        df: pd.DataFrame, config_file: dict, table_name: str, written_type: Literal["append", "replace"]
) -> None:
    """
    Escritura en la database del las tablas del modelo (se usa para entrenamiento y predicción)
    """

    # en el df se incluye una variable temporal -> timestamp para que quede la traza
    df["Date_Timestamp"] = pd.Timestamp.now()

    # se instancia la clase de DB
    db = DataBaseProjectSQLAlchemy(db_name=config_file["database"]["database_name"])
    db.db_connect()  # conecta la database (si no está creada, se crea en el momento)
    db.write_table(df, config_file["output"]["tables"][table_name], written_type)
