from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from statistics import mean


class ModelDependencyError(RuntimeError):
    """Raised when an optional ML dependency is unavailable."""


class Predictor(ABC):
    name: str

    @abstractmethod
    def fit(self, features: list[list[float]], targets: list[float]) -> None:
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

    def fit(self, features: list[list[float]], targets: list[float]) -> None:
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

    def fit(self, features: list[list[float]], targets: list[float]) -> None:
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

    def fit(self, features: list[list[float]], targets: list[float]) -> None:
        estimator_cls = self._load_estimator()
        self.estimator_ = estimator_cls(**self.estimator_kwargs)
        self.estimator_.fit(features, targets)

    def predict(self, history_ms: list[float]) -> float:
        if self.estimator_ is None:
            raise RuntimeError(f"Model '{self.name}' has not been fitted")
        feature_row = [[history_ms[-1], mean(history_ms), min(history_ms), max(history_ms), max(history_ms) - min(history_ms), float(len(history_ms))]]
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
        return NaiveLastValuePredictor()
    if normalized == 'moving_average':
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
        return OptionalBackendPredictor(
            name='lstm',
            dependency_module='tensorflow',
            estimator_path='tensorflow.keras.wrappers.scikit_learn.KerasRegressor',
            estimator_kwargs=params,
        )
    raise ValueError(f'Unsupported model: {name}')
