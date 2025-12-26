from typing import Literal
import pandas as pd

from sqlalchemy import create_engine

from src.utils.utils import get_global_path
from src.utils.logger import get_logger

logger = get_logger()


class DataBaseProjectSQLAlchemy:
    def __init__(self, db_name):

        self.base_path = get_global_path()
        logger.info(f"Directorio donde será creada la databae {self.base_path}")
        self.db_name = db_name
        self.connection = None  # se crea al conectar la database

    def db_connect(self):
        """
        Conectar a la base de datos usando SQLAlchemy.
        """
        db_path = self.base_path / self.db_name
        engine = create_engine(f"sqlite:///{db_path}.db")
        self.connection = engine.raw_connection()
        logger.info(f"Conexión a la database realizada {self.db_name}")

    def write_table(self, df: pd.DataFrame, table_name: str, written_type: Literal["fail", "replace", "append"]):
        """
        Escritura de tabla usando SQLAlchemy.
        """
        if self.connection is not None:
            df.to_sql(table_name, self.connection, if_exists=written_type, index=False)
            logger.info(f"Tabla escrita en la database {self.db_name}")
        else:
            raise RuntimeError("No database connection. You must connect to the database previously")

    def read_table(self, query: str):
        if self.connection is not None:
            data = pd.read_sql(query, self.connection)
            logger.info(f"Lectura de tabla desde la database {self.db_name}")
            return data
        else:
            raise RuntimeError("No database connection. You must connect to the database previously")
