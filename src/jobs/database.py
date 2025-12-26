from src.utils.utils import load_config_file, get_global_path
from src.database.write import write_data_to_db
from src.utils.logger import get_logger

logger = get_logger()


def main():

    logger.info("Inicio Job de creación de la database y tablas asociadas")
    # ruta a añadir (directorio
    config_file_path = "config_file/data.yml"

    base_path = get_global_path()
    # se incluye el directorio donde se encuentran tanto data inputs como ficheros de configuración
    data_path = base_path / "data"
    config_file_path = data_path / config_file_path  # se añade el directorio con el fichero donde está el fichero yaml
    config_file = load_config_file(config_file_path)  # lee fichero de configuración
    write_data_to_db(config_file, data_path)
    logger.info("Job de creación database finalizado")


if __name__ == "__main__":
    main()
