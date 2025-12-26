from typing import Union, List, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import pins

from sklearn.ensemble import RandomForestClassifier
from vetiver import VetiverModel

from src.utils.logger import get_logger
from src.utils.utils import load_config_file

logger = get_logger()


class Predict:

    def __init__(self, config_file_path: Union[Path, str]):

        config_file = load_config_file(config_file_path)
        self.categorical_features = config_file["data"]["categorical_feats"]
        self.pred_threshold = config_file["model"]["final_decision"]["threshold"]

    @staticmethod
    def load_model(
            model_name: Literal["k_neighbors", "random_forest", "tree_decision", "gradient_boosting"] = "random_forest",
            path: str = "artifacts/"):
        """
        Carga del modelo diferenciando si es de scikit-learn
        """
        available_model_type = ["k_neighbors", "random_forest", "decision_tree", "gradient_boosting"]
        if model_name not in available_model_type:
            logger.info(f"Nombre {model_name} no válido. Solo disponible {available_model_type}")
            raise ValueError(f"Nombre {model_name} no válido. Solo disponible {available_model_type}")

        # para cargar el modelo siempre tienes que acceder al directorio
        board = pins.board_folder(path=path, versioned=True, allow_pickle_read=True)
        v = VetiverModel.from_pin(board, name=model_name)  # este es el objeto que guardaremos
        logger.info(f"Cargado el artefacto del en {path} con nombre {model_name}")
        return v.model

    def predict(self, model: RandomForestClassifier, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicción del modelo.
        """
        df = self.__select_categorical_features(df)
        df = self.__check_features(df, model.feature_names_in_)

        pred_df = model.predict_proba(df.drop(columns="id"))[:, 1]
        probs = pd.DataFrame(
            pred_df,
            columns=["Probability"],
            index=df.index
        )
        probs["Prediction"] = np.where(probs["Probability"] > self.pred_threshold, 1, 0)
        pred_df = pd.concat([df, probs], axis=1).reset_index()
        return pred_df[["Date", "id", "Probability", "Prediction"]]

    def __select_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Se tienen en cuenta las variables categóricas.
        """
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
                logger.info(f"La variable {col} es convertida a categórica")

        df.index = df["Date"]
        df = df.drop(columns="Date")
        logger.info("La variable `Date` se convierte a indice del dataframe")

        df = pd.get_dummies(df, drop_first=True, dtype='int')
        logger.info("Las variables categóricas se han convertido a variables dummy")
        return df

    @staticmethod
    def __predictions_according_threshold(y_pred: np.ndarray, pred_threshold: float) -> np.ndarray:
        """
        Uso de la predicción utilizando el umbral adecuado por el usuario
        """
        return (y_pred >= pred_threshold).astype(int)

    @staticmethod
    def __check_features(df: pd.DataFrame, model_features: List[str]):
        """
        Chequea si las variables de entrada del modelo son las mismas que el dataframe
        """
        model_features_check = set(model_features) - set(df.columns)
        for col in model_features_check:
            df[col] = 0

        # se reordena exactamente como espera el modelo
        features = list(model_features)
        features.append("id")
        df = df[features]
        return df
