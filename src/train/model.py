from typing import Dict, Tuple, Union, Optional, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import pins

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from vetiver import VetiverModel, VetiverMeta
from vetiver import vetiver_pin_write

from src.utils.logger import get_logger
from src.utils.utils import load_config_file

logger = get_logger()


class ClassificationTrainer:

    def __init__(
            self,
            config_file_path: Union[Path, str],
            model_type: Literal["k_neighbors", "random_forest", "decision_tree", "gradient_boosting"] = "random_forest"
    ):

        config_file = load_config_file(config_file_path)

        available_model_type = ["k_neighbors", "random_forest", "decision_tree", "gradient_boosting"]
        if model_type not in available_model_type:
            logger.info(f"Nombre {model_type} no válido. Solo disponible {available_model_type}")
            raise ValueError(f"Nombre {model_type} no válido. Solo disponible {available_model_type}")

        self.model_type = model_type
        self.target = config_file["data"]["target"]
        self.categorical_features = config_file["data"]["categorical_feats"]

        self.model_params = config_file["model"]["model_type"][model_type]["model_params"]
        self.training_process = config_file["model"]["training_params"]

        self.model_grid_search = None  # entrenamiento del modelo grid-search (con hiperparámetros y cross-validation)
        self.model = None  # se define una vez entrenado el modelo

    @staticmethod
    def __get_metrics(
            df: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calcula métricas de bondad de ajuste (accuracy, precision, recall y f1-score).
        Puedes utilizar otras métricas de clasificación si lo consideras
        """

        if isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            msg = "El dataframe de predicciones no tiene el formato adecuado. Debe ser Pandas dataFrame"
            logger.error(msg)
            raise TypeError(msg)

        metrics = {
            "accuracy": float(accuracy_score(df["y_true"], df["y_pred"])),
            "precision": float(precision_score(df["y_true"], df["y_pred"])),
            "recall": float(recall_score(df["y_true"], df["y_pred"])),
            "f1": float(f1_score(df["y_true"], df["y_pred"]))
        }
        logger.info("Métricas de bondad de ajuste son calculadas")
        return metrics

    def get_train_val_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Obtiene la muestra de entrenamiento y de validación
        """

        if not isinstance(df, pd.DataFrame):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.error(msg)
            raise TypeError(msg)

        if "id" not in df.columns:
            msg = "`id` ha sido eliminado del dataset. La columna de identificación debe mantenerse"
            logger.error(msg)
            raise ValueError(msg)

        train_df, val_df = self.__get_stratified_sample(df)
        logger.info(f"Muestra de entrenamiento: {train_df.shape[1]}")
        logger.info(f"Muestra de validacion: {val_df.shape[1]}")
        return train_df, val_df

    def __get_stratified_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Dataframe de la muestra de entrenamiento del modelo. Es una extracción estratificada por target
        """
        train_df, val_df = train_test_split(
            df,
            train_size=self.training_process["sample"],
            stratify=df[self.target],
            random_state=self.training_process["seed"]
        )
        return train_df, val_df

    def _previous_check_train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Chequeo formato
        """
        if not isinstance(train_df, pd.DataFrame) or not isinstance(val_df, pd.DataFrame):
            msg = f"Incorrecto tipado del dataset de entrada. Debe ser un dataframe de spark"
            logger.error(msg)
            raise TypeError(msg)

        if self.target not in train_df.columns or self.target not in val_df.columns:
            msg = f"Error. La variable {self.target} no se encuentra en el dataset de entrenamiento o validación"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _check_pred_threshold(pred_threshold: float):
        """
        Analiza si el valor del umbral de predicción es adecuado
        """
        if not 0 <= pred_threshold <= 1:
            msg = "Incorrecto valor para el threshold de toma de decisión del modelo. " \
                  f"Debe estar entre 0 y 1 y toma el valor {pred_threshold}"
            logger.info(msg)
            raise ValueError(msg)

    def train_model(
            self, train_df: pd.DataFrame, val_df: pd.DataFrame, pred_threshold: float = 0.5
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Entrenamiento del modelo de scikit-learn (aplicacióon a un modelo random-forest)
        """
        self._check_pred_threshold(pred_threshold)
        self._previous_check_train_model(train_df, val_df)

        features_model = [col for col in train_df.columns if col not in [self.target, "Date", "id"]]
        logger.info(f"Obtenidas las features del modelo: {features_model}")

        X_train = train_df[features_model].copy()
        y_train = train_df[self.target]
        X_val = val_df[features_model].copy()
        y_val = val_df[self.target]

        logger.info("Identificación variables categóricas para usar en el modelo")
        X_train = self.__select_categorical_features(X_train)
        X_val = self.__select_categorical_features(X_val)

        # X_train = X_train.drop(columns=["id", "Date"])
        # X_val = X_val.drop(columns=["id", "Date"])
        logger.info(f"Entrenamiento de un modelo {self.model_type} usando scikit-learn")
        self.__model_fitted(X_train, y_train)
        y_pred_train = self.model.predict_proba(X_train)[:, 1]
        y_pred_val = self.model.predict_proba(X_val)[:, 1]
        logger.info("Obtención de las predicciones del modelo (train y validación - Cálculo probabilidades")

        y_pred_train = self.__predictions_according_threshold(y_pred_train, pred_threshold)
        y_pred_val = self.__predictions_according_threshold(y_pred_val, pred_threshold)
        logger.info(f"Realización de las predicciones del modelo - Toma decisión con {pred_threshold}")

        predictions_train_df = self.__model_predictions_format(y_train, y_pred_train)
        predictions_val_df = self.__model_predictions_format(y_val, y_pred_val)
        train_metrics = self.__get_metrics(predictions_train_df)
        val_metrics = self.__get_metrics(predictions_val_df)
        logger.info("Obtenidas las métricas del modelo para la muestra de entrenamiento y validación")
        return {"train_sample": train_metrics, "val_sample": val_metrics}

    @staticmethod
    def __predictions_according_threshold(y_pred: np.ndarray, pred_threshold: float) -> np.ndarray:
        """
        Uso de la predicción utilizando el umbral adecuado por el usuario
        """
        return (y_pred >= pred_threshold).astype(int)

    @staticmethod
    def __model_predictions_format(y_real: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Paso de formato a pandas dataframe
        """
        if len(y_real) != len(y_pred):
            msg = "Dimensiones de y_real y y_pred no coinciden"
            logger.error(msg)
            raise ValueError(msg)

        result_df = pd.DataFrame(
            {
                "y_true": y_real.values,  # es una serie obtenemos el array
                "y_pred": y_pred.flatten()  # es un np.ndarray pero por si fuese (n,1)
             }
        )
        return result_df

    def __model_fitted(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrenamiento de un modelo RandomForest con cross-validation a partir de
        la muestra de entrenamiento (feats and target column)
        """

        cv = StratifiedKFold(
            n_splits=self.training_process["cv"], shuffle=True, random_state=self.training_process["seed"]
        )

        estimator = self.__machine_learning_estimator(params=None)
        self.model_grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.model_params,
            scoring=self.training_process["metric"],
            cv=cv,
            n_jobs=1,
            verbose=1
        )

        self.model_grid_search.fit(
            X_train, y_train
        )
        logger.info(f"Entrenamiento del modelo grid-search: {self.model_params}")

        # Entrenamiento del mejor modelo cn toda la muestra
        best_params = self.model_grid_search.best_params_
        logger.info(f"Mejores parámetros encontrados: {best_params}")

        self.model = self.__machine_learning_estimator(params=best_params)
        self.model.fit(X_train, y_train)
        logger.info(f"Entrenamiento final de {self.model_type} con {best_params}")

    def __machine_learning_estimator(self, params: Union[Dict, None] = None) -> Union[
                RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, DecisionTreeClassifier,
            ]:
        """
        Escoge el modelo según `model_type`
        """
        params = {} if params is None else params
        seed = self.training_process["seed"]
        if self.model_type == "random_forest":
            estimator = RandomForestClassifier(**params, random_state=seed)
        elif self.model_type == "decision_tree":
            estimator = DecisionTreeClassifier(**params, random_state=seed)
        elif self.model_type == "k_neighbors":
            estimator = KNeighborsClassifier(**params)
        elif self.model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(**params, random_state=seed)
        return estimator

    def __select_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Se tienen en cuenta las variables categóricas
        """
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
                logger.info(f"La variable {col} es convertida a categórica")
        df = pd.get_dummies(df, drop_first=True, dtype='int')
        logger.info("Las variables categóricas se han convertido a variables dummy")
        return df

    @staticmethod
    def save_model(
            model: Union[
                RandomForestClassifier, GradientBoostingClassifier, KNeighborsClassifier, DecisionTreeClassifier
            ],
            metrics: Optional[Dict[str, Dict[str, float]]] = None,
            model_name: str = "random_forest",
            path: str = "artifacts/",
            data: Optional[pd.DataFrame] = None
    ):
        """
        Guardado del modelo (el directorio se obtiene de los atributos de la clase)
        """

        # pin siempre guarda un versionado específico internamente (es un hash combinado con fecha)
        board = pins.board_folder(path=path, versioned=True, allow_pickle_read=True)

        meta = VetiverMeta(user={"metrics": metrics}).to_dict()
        v = VetiverModel(
            model, model_name=model_name, prototype_data=data, metadata=meta, versioned=True
        )   # este es el objeto que guardaremos
        vetiver_pin_write(board, v)
        logger.info(f"Guardado el artefacto del en {path} con nombre {model_name}")
