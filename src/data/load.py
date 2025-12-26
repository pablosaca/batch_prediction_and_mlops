from typing import Union, Literal

import pandas as pd

from src.database.db_connect import DataBaseProjectSQLAlchemy
from src.utils.utils import get_global_path, load_config_file
from src.utils.logger import get_logger

logger = get_logger()


class LoadData:
    """
    Clase que obtiene los tablones necesarios para realizar el entrenamiento y predicción del modelo
    """

    def __init__(self, job_type: Literal["train", "predict"] = "train"):

        self.job_type = job_type

        # ruta a añadir (directorio)
        config_file_path = "config_file/data.yml"

        # se obtiene la ruta absoluta del script actual
        script_path = get_global_path()

        # se añade el directorio con el fichero donde está el fichero yaml
        config_file_path = script_path / "data" / config_file_path
        self.config_file = load_config_file(config_file_path)  # lee fichero de configuración

    def get_basic_data(self, load_target: bool = True) -> pd.DataFrame:
        """
        Obtención de tablón de features (y target asociado si aplica)
        En este método se cargan los ficheros y se genera el tablón para continuar el proceso
        Debe ser un método con el que se disponga de un macrotablón para realizar las diferentes operativas:
        - EDA y modelización
        - Predicción del modelo ante una simulación productiva del modelo
        """

        # conexión a la database creada en el momento en que se instancia la clase (se llama también al método)
        database = DataBaseProjectSQLAlchemy(db_name=self.config_file["database"]["database_name"])
        database.db_connect()

        dataframes = {}
        for table_name in self.config_file["database"]["tables_name"].values():
            if not isinstance(table_name, str):
                for table_aux_name in table_name.values():
                    query = f"""SELECT * FROM {table_aux_name}"""
                    dataframes[table_aux_name] = database.read_table(query)
                    logger.info(f"Cargada tabla {table_aux_name}")
            else:
                if table_name == "xsell_target" and not load_target:
                    continue
                else:
                    query = f"""SELECT * FROM {table_name}"""
                    dataframes[table_name] = database.read_table(query)
                    logger.info(f"Cargada tabla {table_name}")

        profile_table_name_df = dataframes["xsell_profile"]

        date = self.config_file["output"]["training_date_sample"]
        if self.job_type == "train":
            profile_table_name_df = profile_table_name_df[profile_table_name_df["Date"] < date]
        else:
            profile_table_name_df = profile_table_name_df[profile_table_name_df["Date"] >= date]

        performance_table_name_df = dataframes["xsell_performance"]
        socioeconomic_table_name_df = dataframes["xsell_socioeconomic"]

        df = self.__merge_tables(
            profile_table_name_df, performance_table_name_df, "id", "left"
        )
        df = self.__merge_tables(
            df, socioeconomic_table_name_df, "Region_Code", "left"
        )
        logger.info("Obtención del tablón de features")

        if load_target:
            df = self.__merge_tables(df, dataframes["xsell_target"], "id", "left")
            logger.info("Obtención de tablón con target")
        return df

    @staticmethod
    def __merge_tables(
            df1: pd.DataFrame, df2: pd.DataFrame, cols_merge: Union[str, list], how: Literal["left", "inner"]
    ) -> pd.DataFrame:
        """
        Cruce de tablas para construir el macrotablón
        """
        logger.info(f"Se cruzan dos tablas ({how}-join) por {cols_merge}")
        df1 = df1.merge(df2, on=cols_merge, how=how)
        return df1
