import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from keras import Sequential
from keras.callbacks import History, EarlyStopping, TensorBoard
from keras.layers import LSTM, Activation, Dense
from keras.utils import plot_model

from ikfs_anomaly_detector.core.custom_types import Signals
from ikfs_anomaly_detector.core.format.telemetry import TelemetryAttrs
from ikfs_anomaly_detector.core.utils import PROJECT_PATH, fill_zeros_with_previous
from ikfs_anomaly_detector.intellectual.utils import (
    calculate_covariance_matrix,
    mahalanobis_distance,
    z_normalization,
)

SIGNALS_FOR_TRAINING = (
    TelemetryAttrs.ppt_ripple,
    TelemetryAttrs.ppt_sample_count,
    TelemetryAttrs.scanner_angle,
    TelemetryAttrs.str_power,
    TelemetryAttrs.tu1_temperature,
    TelemetryAttrs.tu2_temperature,
)


@dataclass
class PredictorResult:
    signal: str  # Название сигнала
    data: np.ndarray  # Значения сигнала
    predicted_data: np.ndarray  # Предсказанные значения сигнала
    mahalanobis_distance: np.ndarray  # Расстояние Махаланобиса


class LSTMPredictor:
    LOOKBACK_SEQUENCE_LENGTH = 16
    PREDICTED_SEQUENCE_LENGTH = 4

    EPOCHS = 100
    MIN_EPOCHS = 1
    MIN_DELTA = 0.002
    BATCH_SIZE = 100
    VALIDATION_SPLIT = 0.05

    def __init__(self, models_dir: str = '', tensorboard_dir: str = '') -> None:
        self._models_dir = models_dir or os.path.join(PROJECT_PATH, 'models')
        self._tensorboard_dir = tensorboard_dir

        self._model = Sequential([
            LSTM(
                input_shape=(self.LOOKBACK_SEQUENCE_LENGTH, 1),
                units=32,
                return_sequences=True,
            ),
            Activation('relu'),
            LSTM(
                units=32,
            ),
            Activation('relu'),
            Dense(units=self.PREDICTED_SEQUENCE_LENGTH),
        ])

    def train(self, signal_name: str, normal_data: np.ndarray) -> History:
        print(f'Обучение LSTM-предиктора для сигнала "{signal_name}"...')

        x_train, y_train = self._create_subsequences(self._preprocess(normal_data))
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # samples, sample_len, features

        self._model.compile(optimizer='adam', loss='mse')

        callbacks = [
                EarlyStopping('val_loss', patience=self.MIN_EPOCHS, min_delta=self.MIN_DELTA),
        ]
        if self._tensorboard_dir:
            os.makedirs(self._tensorboard_dir, exist_ok=True)
            callbacks.append(
                TensorBoard(
                    log_dir=self._get_tensorboard_logs_dir(signal_name),
                    batch_size=self.BATCH_SIZE,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=True,
                    write_images=True,
                ))

        history = self._model.fit(
            x_train, y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_split=self.VALIDATION_SPLIT,
            shuffle=True,
            callbacks=callbacks,
        )

        models_path = self._get_model_path(signal_name)
        os.makedirs(os.path.dirname(models_path), exist_ok=True)
        self._model.save_weights(models_path)
        print(f'Модель сохранена в "{models_path}"')

        return history

    def analyze(self, signal: Signals) -> PredictorResult:
        assert len(signal) == 1, 'Allowed to analyze only 1 singal'

        signal_name, data = next(iter(signal.items()))

        model_path = self._get_model_path(signal_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model for {signal_name} is not found. Please train')
        self._model.load_weights(model_path)

        data = self._preprocess(data)
        subsequences, next_values = self._create_subsequences(data)
        subsequences = np.reshape(subsequences, (subsequences.shape[0], subsequences.shape[1], 1))

        predicted_next_values = self._model.predict(subsequences, batch_size=self.BATCH_SIZE)
        errors = next_values - predicted_next_values
        error_mean, error_cov = calculate_covariance_matrix(errors)

        return PredictorResult(
            signal=signal_name,
            data=data,
            predicted_data=predicted_next_values,
            mahalanobis_distance=np.array([
                mahalanobis_distance(e_vec, error_mean, error_cov)
                for e_vec in errors
            ])
        )

    def plot_model(self, img_path: str, show_shapes=True, show_layer_names=True) -> None:
        plot_model(
            self._model,
            to_file=img_path,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
        )

    def _get_model_path(self, signal_name: str) -> str:
        return os.path.join(self._models_dir, 'predictor', f'{signal_name}.h5')

    def _get_tensorboard_logs_dir(self, signal_name: str) -> str:
        return os.path.join(self._tensorboard_dir, 'predictor', signal_name)

    @staticmethod
    def _preprocess(data: np.ndarray) -> np.ndarray:
        return z_normalization(fill_zeros_with_previous(data))

    @staticmethod
    def _create_subsequences(
            series: np.ndarray,
            previous_count: int = LOOKBACK_SEQUENCE_LENGTH,
            next_count: int = PREDICTED_SEQUENCE_LENGTH,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert previous_count > next_count

        previous_values, next_values = [], []
        for i in range(previous_count, len(series) - next_count):
            previous_values.append(series[i - previous_count:i])
            next_values.append(series[i:i + next_count])

        return np.array(previous_values), np.array(next_values)
