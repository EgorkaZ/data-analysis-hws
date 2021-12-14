import numpy as np

def add_dim_to_middle(arr: np.ndarray) -> np.ndarray:
  return np.reshape(arr, (arr.shape[0], 1, arr.shape[-1]))

def take_dim_from_middle(arr: np.ndarray) -> np.ndarray:
  return np.asarray([arr[i][0] for i in range(0, len(arr))])