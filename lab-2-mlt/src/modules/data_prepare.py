import copy
import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple

from modules.util import *


class ScaledData:
  scaler: MinMaxScaler
  normalized: np.ndarray
  denormalized: np.ndarray

  def nrm(self) -> np.ndarray:
    return self.normalized
  def den(self) -> np.ndarray:
    return self.denormalized


def from_normalized(normalized: np.ndarray, scaler: MinMaxScaler) -> ScaledData:
  sd = ScaledData()
  sd.scaler = scaler
  sd.normalized = copy.deepcopy(normalized)
  sd.denormalized = scaler.inverse_transform(normalized.reshape(-1, 1)).reshape(normalized.shape)
  return sd

def from_denormalized(denormalized: np.ndarray, scaler: MinMaxScaler) -> ScaledData:
  sd = ScaledData()
  sd.scaler = scaler
  sd.denormalized = copy.deepcopy(denormalized)

  sd.normalized = scaler.transform(denormalized.reshape(-1, 1)).reshape(denormalized.shape)
  return sd

class SplittedData:
  train: ScaledData
  test: ScaledData

  def __init__(self, data: np.ndarray, scaler: MinMaxScaler, test_ratio: float):
    pos = int(round(len(data) * (1-test_ratio)))
    self.train = from_denormalized(data[:pos], scaler)
    self.test = from_denormalized(data[pos:], scaler)


def extract_dataset(dataset: np.ndarray, look_back: int, look_forward: int) -> Tuple[np.ndarray, np.ndarray]:
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-look_forward):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[(i + look_back):(i + look_back + look_forward), 0])
  return np.array(dataX), np.array(dataY)

def split_and_reshape(dataset: np.ndarray, scaler: MinMaxScaler, look_back: int, look_forward: int, test_ratio: float) -> Tuple[SplittedData, SplittedData]:
  X, y = extract_dataset(dataset, look_back, look_forward)
  print("X:{},y:{}".format(X.shape, y.shape))

  X_sd = SplittedData(X, scaler, test_ratio)
  y_sd = SplittedData(y, scaler, test_ratio)
  return X_sd, y_sd

def extract_and_scale_data(dataset: pd.DataFrame, column_to_predict: int) -> Tuple[np.ndarray, MinMaxScaler]:
  # Selection of variables X and Y
  dataset = dataset.iloc[:,column_to_predict].to_numpy()
  dataset = dataset.reshape(-1, 1)
  dataset = dataset.astype("float32")
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaler = scaler.fit(dataset)
  return dataset, scaler

