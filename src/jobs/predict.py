from typing import Literal

from src.data.load import LoadData
from src.predict.predict import Predict
from src.preprocessing.preprocessing import load_imputation_process, impute_nulls_for_numerical_cols_out_sample
from src.database.write import write_model_data_to_db
from src.utils.utils import load_config_file, get_global_path
from src.utils.logger import get_logger

logger = get_logger()


def main():
    logger.info("Inicio Job de predicción del modelo")

    load_data = LoadData(job_type="predict")
    final_df = load_data.get_basic_data(load_target=False)
    final_df = final_df.drop_duplicates(["id"])  # por si hay duplicados

    # carga de los artefactos
    preprocessing_values_dict = load_imputation_process(path="../../artifacts", file_name="preprocessing")

    base_path = get_global_path()
    # se incluye el directorio donde se encuentra el fichero de configuración de la modelización
    path = base_path / "data" / "config_file" / "model_params.yml"

    model_type: Literal["k_neighbors", "random_forest", "decision_tree", "gradient_boosting"] = "random_forest"
    predict_model = Predict(path)
    model = predict_model.load_model(model_name=model_type, path="../../artifacts")

    # imputación de valores faltantes
    final_df = impute_nulls_for_numerical_cols_out_sample(
        final_df,
        "Annual_Premium",
        preprocessing_values_dict["Annual_Premium"]
    )
    final_df = impute_nulls_for_numerical_cols_out_sample(
        final_df, "Age", preprocessing_values_dict, "Gender"
    )

    # ejecución del modelo
    predictions_df = predict_model.predict(model, final_df)

    # guardado tablas en database:
    config_file = load_data_config_file()
    write_model_data_to_db(
        df=predictions_df, config_file=config_file, table_name="predictions", written_type="replace"
    )
    logger.info("Job de predicción finalizado")


def load_data_config_file():
    config_file_path = "config_file/data.yml"
    base_path = get_global_path()
    # se incluye el directorio donde se encuentran tanto data inputs como ficheros de configuración
    data_path = base_path / "data"
    config_file_path = data_path / config_file_path  # se añade el directorio con el fichero donde está el fichero yaml
    config_file = load_config_file(config_file_path)  # lee fichero de configuración
    return config_file


if __name__ == "__main__":
    main()
