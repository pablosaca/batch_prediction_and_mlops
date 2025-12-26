from typing import Optional, Union, Tuple, Dict

import pandas as pd
import pins

from src.utils.logger import get_logger

logger = get_logger()


def impute_nulls_for_numeric_cols(
    df: pd.DataFrame,
    method_name: str,
    col_name: str,
    stratific_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Imputa valores nulos en columnas numéricas con media o mediana,
    de forma global o estratificada por una columna.
    """
    if stratific_col is None:
        if method_name == "mean":
            value = df[col_name].mean()
        elif method_name == "median":
            value = df[col_name].median()
        else:
            msg = f"No disponible el método {method_name} para la imputación de {col_name}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Calculo {method_name} para imputación: {col_name} → {value}")
        df[col_name] = df[col_name].fillna(value)
        value_dict = {col_name: value}

    elif stratific_col in df.columns:
        if method_name == "mean":
            value_map = df.groupby(stratific_col)[col_name].mean().round()
        elif method_name == "median":
            value_map = df.groupby(stratific_col)[col_name].median()
        else:
            msg = f"No disponible el método {method_name} para la imputación de {col_name}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Calculo {method_name} estratificado para {col_name}: {value_map.to_dict()}")
        df[col_name] = df.apply(
            lambda row: value_map[row[stratific_col]] if pd.isna(row[col_name]) else row[col_name],
            axis=1
        )
        value_dict = {col_name: value_map.to_dict()}
    else:
        msg = f"{stratific_col} no es una columna de la tabla de partida. Revisa el nombre de columnas {df.columns}"
        logger.info(msg)
        raise ValueError(msg)
    return df, value_dict


def impute_nulls_for_numerical_cols_out_sample(
    df: pd.DataFrame,
    col_name: str,
    impute_value_or_mapping: Union[float, int, Dict[Union[str, int], Union[float, int]]],
    stratific_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Imputación de valores para predicciones futuras.
    Puede usar imputación simple o estratificada por columna.
    """
    if isinstance(impute_value_or_mapping, (float, int)):
        # Imputación simple
        df[col_name] = df[col_name].fillna(impute_value_or_mapping)

    elif isinstance(impute_value_or_mapping, dict):
        logger.info(f"Imputando {col_name} usando {stratific_col}")
        if stratific_col is None:
            raise ValueError("Para imputación estratificada, se requiere 'stratific_col'.")

        # Mapeo estratificado (dict: valor_estrato → valor_imputación)
        df[col_name] = df[col_name].fillna(
            df[stratific_col].map(impute_value_or_mapping["Age"])
        )

    logger.info(f"{col_name} ha sido imputada según {impute_value_or_mapping}")
    return df


def save_imputation_process(
        task: Dict[str, Union[float, Dict[str, float]]],
        file_name: str = "preprocessing",
        path: str = "artifacts/") -> None:
    """
    Guardado del diccionario para obtener el proceso de imputación de valores nulos
    """
    # pin siempre guarda un versionado específico internamente (es un hash combinado con fecha)
    board = pins.board_folder(path=path)
    board.pin_write(task, name=file_name, type="json")
    logger.info(f"Guardado el artefacto del preprocesado en {path} con nombre {file_name}")


def load_imputation_process(
        file_name: str = "preprocessing",
        path: str = "artifacts/",
        file_version: Optional[str] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Carga del diccionario para obtener el proceso de imputación de valores nulos
    """
    board = pins.board_folder(path=path)
    loaded = board.pin_read(name=file_name, version=file_version)
    return loaded
