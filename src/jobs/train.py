from typing import Literal
import pandas as pd

from src.data.load import LoadData
from src.train.model import ClassificationTrainer
from src.preprocessing.preprocessing import (
    impute_nulls_for_numeric_cols,
    impute_nulls_for_numerical_cols_out_sample,
    save_imputation_process
)
from src.database.write import write_model_data_to_db
from src.utils.utils import load_config_file, get_global_path
from src.utils.logger import get_logger

logger = get_logger()


def main():

    logger.info("Inicio Job de entrenamiento del modelo")

    load_data = LoadData()
    final_df = load_data.get_basic_data()
    final_df = final_df.drop_duplicates(["id"])

    # check valores nulos en el target -> si los hay se eliminan
    final_df = null_values_target_check(final_df)

    base_path = get_global_path()
    # se incluye el directorio donde se encuentra el fichero de configuración de la modelización
    path = base_path / "data" / "config_file" / "model_params.yml"

    model_type: Literal["k_neighbors", "random_forest", "decision_tree", "gradient_boosting"] = "k_neighbors"
    model = ClassificationTrainer(config_file_path=path, model_type=model_type)
    train_df, val_df = model.get_train_val_sample(final_df)

    # imputación de valores nulos
    train_df, preprocessing_values_for_annual_premium_dict = impute_nulls_for_numeric_cols(
        train_df, "median", "Annual_Premium"
    )
    train_df, preprocessing_values_for_age_dict = impute_nulls_for_numeric_cols(
        train_df, "mean", "Age", "Gender"
    )

    # actualizar el diccionario
    preprocessing_values_dict = preprocessing_values_for_annual_premium_dict.copy()
    preprocessing_values_dict.update(preprocessing_values_for_age_dict)

    val_df = impute_nulls_for_numerical_cols_out_sample(
        val_df,
        "Annual_Premium",
        preprocessing_values_dict["Annual_Premium"]
    )
    val_df = impute_nulls_for_numerical_cols_out_sample(
        val_df, "Age", preprocessing_values_dict, "Gender"
    )

    metrics = model.train_model(train_df, val_df, 0.12)
    logger.info(f"Las métricas son: {metrics}")

    # guardado
    save_imputation_process(
        task=preprocessing_values_dict, path="../../artifacts"
    )
    model.save_model(
        model=model.model, metrics=metrics, model_name=model_type, path="../../artifacts", data=train_df.head(1)
    )

    # guardado tablas en database:
    config_file = load_data_config_file(config_file_path="config_file/data.yml", folder_name="data")
    write_model_data_to_db(df=train_df, config_file=config_file, table_name="train_data", written_type="append")
    write_model_data_to_db(df=val_df, config_file=config_file, table_name="val_data", written_type="append")
    del train_df, val_df

    logger.info("Job de entrenamiento finalizado")


def load_data_config_file(config_file_path: str, folder_name: str):
    """
    Carga del fichero de configuración para datos
    """

    base_path = get_global_path()
    # se incluye el directorio donde se encuentran tanto data inputs como ficheros de configuración
    data_path = base_path / folder_name
    config_file_path = data_path / config_file_path  # se añade el directorio con el fichero donde está el fichero yaml
    config_file = load_config_file(config_file_path)  # lee fichero de configuración
    return config_file


def null_values_target_check(df: pd.DataFrame, target_col: str = "Response") -> pd.DataFrame:
    """
    Función que chequea valores nulos en la variable target
    """

    logger.info(f"Distribución target: {df[target_col].value_counts(normalize=True, dropna=False)}")
    null_values_filter = df[target_col].isna()
    df_ = df[~null_values_filter]
    if not df_.empty:  # chequeamos si el dataframe de pandas está vacío
        logger.info(f"Eliminación valores nulos {null_values_filter.sum()}")
        return df_
    else:
        return df


if __name__ == "__main__":
    main()
