from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from statistics import mean

from ml_bsr.features import build_feature_vector


class ModelDependencyError(RuntimeError):
    """Raised when an optional ML dependency is unavailable."""


def _validate_params(name: str, params: dict[str, object], allowed: set[str]) -> None:
    unexpected = set(params) - allowed
    if unexpected:
        raise ValueError(f"Unsupported parameters for '{name}': {sorted(unexpected)}")


class Predictor(ABC):
    name: str

    @abstractmethod
    def fit(self, histories: list[list[float]], targets: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, history_ms: list[float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass
class NaiveLastValuePredictor(Predictor):
    name: str = "naive_last_value"
    fitted_: bool = False

    def fit(self, histories: list[list[float]], targets: list[float]) -> None:
        self.fitted_ = True

    def predict(self, history_ms: list[float]) -> float:
        if not history_ms:
            raise ValueError("history_ms must not be empty")
        return float(history_ms[-1])

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "fitted": self.fitted_}


@dataclass
class MovingAveragePredictor(Predictor):
    window: int = 3
    name: str = "moving_average"
    fitted_: bool = False

    def fit(self, histories: list[list[float]], targets: list[float]) -> None:
        self.fitted_ = True

    def predict(self, history_ms: list[float]) -> float:
        if not history_ms:
            raise ValueError("history_ms must not be empty")
        window = history_ms[-self.window:] if self.window > 0 else history_ms
        return float(mean(window))

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "window": self.window, "fitted": self.fitted_}


@dataclass
class OptionalBackendPredictor(Predictor):
    name: str
    dependency_module: str
    estimator_path: str
    estimator_kwargs: dict[str, object] = field(default_factory=dict)
    estimator_: object | None = None

    def _load_estimator(self) -> type:
        try:
            module_name, attribute = self.estimator_path.rsplit('.', 1)
            import_module(self.dependency_module)
            module = import_module(module_name)
            return getattr(module, attribute)
        except Exception as exc:
            raise ModelDependencyError(
                f"Model '{self.name}' requires optional dependency '{self.dependency_module}'."
            ) from exc

    def fit(self, histories: list[list[float]], targets: list[float]) -> None:
        estimator_cls = self._load_estimator()
        self.estimator_ = estimator_cls(**self.estimator_kwargs)
        features = [build_feature_vector(history) for history in histories]
        self.estimator_.fit(features, targets)

    def predict(self, history_ms: list[float]) -> float:
        if self.estimator_ is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted")
        feature_row = [build_feature_vector(history_ms)]
        value = self.estimator_.predict(feature_row)[0]
        return float(value)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dependency_module": self.dependency_module,
            "estimator": self.estimator_path,
            "fitted": self.estimator_ is not None,
        }


def build_model(name: str, **params: object) -> Predictor:
    normalized = name.lower()
    if normalized == 'naive_last_value':
        _validate_params('naive_last_value', params, set())
        return NaiveLastValuePredictor()
    if normalized == 'moving_average':
        _validate_params('moving_average', params, {'window'})
        return MovingAveragePredictor(window=int(params.get('window', 3)))
    if normalized == 'svr':
        return OptionalBackendPredictor(
            name='svr',
            dependency_module='sklearn',
            estimator_path='sklearn.svm.SVR',
            estimator_kwargs=params,
        )
    if normalized == 'random_forest':
        return OptionalBackendPredictor(
            name='random_forest',
            dependency_module='sklearn',
            estimator_path='sklearn.ensemble.RandomForestRegressor',
            estimator_kwargs=params,
        )
    if normalized == 'xgboost':
        return OptionalBackendPredictor(
            name='xgboost',
            dependency_module='xgboost',
            estimator_path='xgboost.XGBRegressor',
            estimator_kwargs=params,
        )
    if normalized == 'lstm':
        _validate_params('lstm', params, {'epochs', 'batch_size'})
        return TensorFlowLSTMPredictor(**params)
    raise ValueError(f'Unsupported model: {name}')


@dataclass
class TensorFlowLSTMPredictor(Predictor):
    epochs: int = 5
    batch_size: int = 8
    name: str = "lstm"
    model_: object | None = None
    history_length_: int | None = None

    def fit(self, histories: list[list[float]], targets: list[float]) -> None:
        try:
            import numpy as np
            import tensorflow as tf
        except Exception as exc:
            raise ModelDependencyError("Model 'lstm' requires optional dependency 'tensorflow'.") from exc

        if not histories:
            raise ValueError("histories must not be empty")
        self.history_length_ = len(histories[0])
        x_train = np.array(histories, dtype="float32").reshape(len(histories), self.history_length_, 1)
        y_train = np.array(targets, dtype="float32")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.history_length_, 1)),
                tf.keras.layers.LSTM(16),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model_ = model

    def predict(self, history_ms: list[float]) -> float:
        if self.model_ is None or self.history_length_ is None:
            raise RuntimeError("Model 'lstm' has not been fitted")
        if not history_ms:
            raise ValueError("history_ms must not be empty")
        try:
            import numpy as np
        except Exception as exc:
            raise ModelDependencyError("Model 'lstm' requires optional dependency 'numpy'.") from exc
        sequence = history_ms[-self.history_length_:]
        if len(sequence) < self.history_length_:
            sequence = ([sequence[0]] * (self.history_length_ - len(sequence))) + sequence
        batch = np.array(sequence, dtype="float32").reshape(1, self.history_length_, 1)
        return float(self.model_.predict(batch, verbose=0)[0][0])

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "fitted": self.model_ is not None,
        }
