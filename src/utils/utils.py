from typing import Union
from pathlib import Path

import yaml


def load_config_file(filepath: Union[Path, str] = "data/config_file/data.yml"):
    """
    Carga fichero de configuración
    """
    with open(filepath, "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


def get_global_path() -> Path:
    """
    Obtiene el directorio donde se encuentra el proyecto
    """

    # se obtiene la ruta absoluta del script actual
    # script_path = Path(__file__).resolve()
    # solo queremos el directorio global
    # base_path = next(p for p in script_path.parents if p.name == "batch_prediction_and_mlops")
    # return base_path
    return Path(__file__).resolve().parent.parent.parent
