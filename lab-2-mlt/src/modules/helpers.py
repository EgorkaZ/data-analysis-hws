import math
import matplotlib.pyplot as plt
import numpy as np
from keras.engine.training import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from typing import Tuple

from modules.util import *
import modules.data_prepare as prepare


def prepare_plot(title: str, xlabel: str = "", ylabel: str = "") -> None:
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

def measure(
    predicted: prepare.ScaledData,
    expected: prepare.ScaledData) -> Tuple[np.ndarray, float]:
  rmse_score = math.sqrt(mean_squared_error(expected.den(), predicted.den()))
  return predicted.den(), rmse_score

def predict_and_measure(
    model: Model,
    data: prepare.ScaledData,
    expected: prepare.ScaledData,
    scaler: MinMaxScaler) -> Tuple[np.ndarray, float]:
  pred_data = add_dim_to_middle(data.nrm()) # dim insertion guard start
  prediction = model.predict(pred_data)
  prediction = take_dim_from_middle(prediction)
  prediction = prepare.from_normalized(prediction, scaler) # dim insertion guard end
  return measure(prediction, expected)


def create_model(
    train_data: prepare.ScaledData,
    test_data: prepare.ScaledData,
    look_back: int,
    epochs: int,
    batch_size: int) -> Model:
  model = Sequential()
  model.add(LSTM(4, return_sequences = True, input_shape=(1, look_back)))
  model.add(Dense(test_data.nrm().shape[-1]))
  model.compile(loss='mean_squared_error', optimizer='adam')
  # add dim guard start
  local_train_data = add_dim_to_middle(train_data.nrm())
  local_test_data = add_dim_to_middle(test_data.nrm())
  model.fit(local_train_data, local_test_data, epochs=epochs, batch_size=batch_size)
  assert(train_data.nrm().ndim == 2)
  assert(test_data.nrm().ndim == 2)
  # add dim guard end
  return model
