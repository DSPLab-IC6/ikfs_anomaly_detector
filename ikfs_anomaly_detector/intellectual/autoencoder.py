import os
from dataclasses import dataclass
from typing import List, Optional

from keras import Sequential
from keras import optimizers
from keras.callbacks import History, EarlyStopping, TensorBoard
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, np
from keras.utils import plot_model

from ikfs_anomaly_detector.core.custom_types import Signals
from ikfs_anomaly_detector.core.utils import fill_zeros_with_previous, PROJECT_PATH
from ikfs_anomaly_detector.intellectual.utils import z_normalization, squared_error, ewma


@dataclass
class SignalsGroup:
    name: str
    signals: List[str]
    signals_data: Optional[Signals] = None


@dataclass
class AutoencoderResult:
    signal_group_name: str
    signals: np.ndarray
    decoded_signals: np.ndarray
    mse: np.ndarray  # Mean squared error
    ewma_mse: np.ndarray


class LSTMAutoencoder:
    EPOCHS = 100
    MIN_EPOCHS = 1
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.05

    EWMA_WINDOW_SIZE = 120
    EWMA_ALPHA = 1 - np.exp(-np.log(2) / EWMA_WINDOW_SIZE)

    def __init__(self, signals_count: int, models_dir: str = '', tensorboard_dir: str = '') -> None:
        self._signals_count = signals_count
        self._models_dir = models_dir or os.path.join(PROJECT_PATH, 'models')
        self._tensorboard_dir = tensorboard_dir

        self._model = Sequential([
            LSTM(units=64, activation='relu', input_shape=(signals_count, 1)),
            RepeatVector(signals_count),
            LSTM(units=64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(1)),
        ])

    def train(self, signals_group: SignalsGroup, clipnorm: float = 0.) -> History:
        if len(signals_group.signals_data) != self._signals_count:
            raise ValueError(f'Модель может обработать строго {self._signals_count} сигналов')

        print(f'Обучение LSTM-автокодировщика для группы сигналов '
              f'"{signals_group.name}" {signals_group.signals}...')

        x_train = np.column_stack([
            self._preprocess(signal)
            for signal in signals_group.signals_data.values()
        ])
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # samples, sample_len, features

        optimizer = optimizers.Adam(clipnorm=clipnorm)
        self._model.compile(optimizer=optimizer, loss='mse')

        callbacks = [
            EarlyStopping('val_loss', patience=self.MIN_EPOCHS, min_delta=0.05),
        ]
        if self._tensorboard_dir:
            os.makedirs(os.path.dirname(self._tensorboard_dir), exist_ok=True)
            callbacks.append(
                TensorBoard(
                    log_dir=self._get_tensorboard_logs_dir(signals_group.name),
                    batch_size=self.BATCH_SIZE,
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=True,
                    write_images=True,
                ))

        history = self._model.fit(
            x_train, x_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_split=self.VALIDATION_SPLIT,
            shuffle=True,
            callbacks=callbacks,
        )

        models_path = self._get_model_path(signals_group.name)
        os.makedirs(os.path.dirname(models_path), exist_ok=True)
        self._model.save_weights(models_path)
        print(f'Модель сохранена в "{models_path}"')

        return history

    def analyze(self, signals_group: SignalsGroup) -> AutoencoderResult:
        if len(signals_group.signals) != self._signals_count:
            raise ValueError(f'Модель может обработать строго {self._signals_count} сигналов')

        model_path = self._get_model_path(signals_group.name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Модель для группы сигналов {signals_group.name} не найдена. Выполните обучение')
        self._model.load_weights(model_path)

        data = np.array([self._preprocess(signal) for signal in signals_group.signals_data.values()])
        data_stacked = np.column_stack(data)
        # samples, sample_len, features
        data_reshaped = data_stacked.reshape(data_stacked.shape[0], data_stacked.shape[1], 1)

        decoded_data_reshaped = self._model.predict(data_reshaped, batch_size=self.BATCH_SIZE)
        decoded_data = decoded_data_reshaped.reshape(data_stacked.shape)
        decoded_data = np.column_stack(decoded_data)

        mse = np.sum([
            squared_error(predictions, targets)
            for predictions, targets in zip(decoded_data, data)
        ], axis=0)
        ewma_mse = ewma(mse, window=self.EWMA_WINDOW_SIZE, alpha=self.EWMA_ALPHA)

        return AutoencoderResult(
            signal_group_name=signals_group.name,
            signals=data,
            decoded_signals=decoded_data,
            mse=mse,
            ewma_mse=ewma_mse,
        )

    def plot_model(self, img_path: str, show_shapes=True, show_layer_names=True) -> None:
        plot_model(
            self._model,
            to_file=img_path,
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
        )

    @staticmethod
    def _preprocess(data: np.ndarray) -> np.ndarray:
        return z_normalization(fill_zeros_with_previous(data))

    def _get_model_path(self, group_name: str) -> str:
        return os.path.join(self._models_dir, 'autoencoder', f'{group_name}.h5')

    def _get_tensorboard_logs_dir(self, group_name: str) -> str:
        return os.path.join(self._tensorboard_dir, 'autoencoder', group_name)
