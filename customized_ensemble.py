import os
import numpy as np
from scipy.stats import mode
from keras.models import Model
from keras.layers import Reshape
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Concatenate
from keras.layers.core import Flatten, Dense

###Only trainable for individual architectures
class CustomizedEnsemble:
  def __init__(self, models : list, voting : str = "hard", verbose : int = 2, save_to = False):
    self.models = models
    self.voting = voting
    self.verbose = verbose
    self.mainModels = []
    self.compParams = None
    self.layerNums = []
    if save_to:
      self.path = save_to

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Train the model by giving the parameters as a dictionary
  def fit(self, params : dict) -> None:
    for i, model in enumerate(self.models):
      if i % self.verbose == 0 or i == 0:
        os.system("cls")
      print(f"Fitting model number {i}. . .")
      model = self.create(model, i)
      model.compile(**self.compParams[i])
      self.layerNums.append(len(model.layers))
      model.fit(**params[i])
      self.mainModels.append(model)

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Compile the model by the parameters as a dictionary
  def compile(self, params : dict) -> None:
    self.compParams = params

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Changes layer names to avoid repeated layer names
  def create(self, model, pos : int):
    base = Sequential()
    for i, layer in enumerate(model.layers):
      layer._name = f"{''.join([i for i in layer._name if not i.isdigit()])}{pos}{i}"
      base.add(layer)
    return base

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Connect all the individual models into one
  def aggregate(self) -> None:
    for i, model in enumerate(self.mainModels):
      if i == 0:
        continue
      elif i == 1:
        pre = self.mainModels[i-1]
        final = Sequential()
        for layer in pre.layers:
          final.add(layer)
        final.add(Flatten())
        curr = model.input
        curr_shape = [i for i in list(curr.shape) if i is not None]
        flatten_shape = np.prod(curr_shape)
        final.add(Dense(flatten_shape))
        final.add(Reshape(tuple(curr_shape)))
        for layer in model.layers:
          final.add(layer)
      else:
        final.add(Flatten())
        curr = model.input
        curr_shape = [i for i in list(curr.shape) if i is not None]
        flatten_shape = np.prod(curr_shape)
        final.add(Dense(flatten_shape))
        final.add(Reshape(tuple(curr_shape)))
        for layer in model.layers:
          final.add(layer)
    if os.path.exists(self.path):
      with open(self.path+"var.txt", "w") as f:
        f.write(" ".join([str(num) for num in self.layerNums])+f" {self.voting}")
        f.close()
      final.save(self.path+"model.h5")

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Make predictions
  @staticmethod
  def predict(models : list, sample : np.ndarray, extra : list) -> int:
    predictions = []
    preds = []
    pred = None
    for model in models:
      predictions.append(model.predict(sample))
    if extra[-1] == "hard":
      for prediction in predictions:
        preds.append(np.argmax(prediction))
      pred = mode(np.array(preds)).mode[0]
    elif extra[-1] == "soft":
      pred = np.sum(predictions, axis=0)
      pred = np.argmax(pred)
    return pred

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Model evaluation
  @staticmethod
  def evaluate(models, X_test : np.ndarray, y_test : np.ndarray, extra : list, batch_size : int, loss_func = None) -> list:
    losses, accs = [], []
    to_bin = False
    if y_test.ndim > 1:
      to_bin = True
    for i in range(0, len(X_test), batch_size):
      batch = X_test[i:i+batch_size]
      batch_y = y_test[i:i+batch_size]
      preds = [np.argmax(np.sum(np.array([model.predict(sample.reshape(-1, *X_test.shape[1:])) for model in models]), axis=0)) for sample in batch]
      if to_bin:
        batch_y = np.argmax(batch_y, axis=1)
      error = batch_y - preds
      acc = (len(batch_y) - np.count_nonzero(error)) / len(batch_y) 
      loss = 0.5 * np.sum(error**2)
      accs.append(acc)
      losses.append(loss)
    mean = np.array([accs, losses]).mean(axis=1)
    print(f"Accuracy : {mean[0]}\nLoss : {mean[1]}")
    return list(mean)

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Disconnect the model back to various models
  @staticmethod
  def decompose(model, extra : list) -> list:
    models = []
    spliter = 0
    layers = model.layers
    for i in range(len(extra)-1):
      model = Sequential()
      indv_layers = layers[spliter:spliter+extra[i]]
      for layer in indv_layers:
        model.add(layer)
      models.append(model)
      spliter += extra[i] + 3
    return models

#-------------------------------------------------------------------------------------------------------------------------------------------------|

  #Load the model and some extra values
  @staticmethod
  def load(path : str) -> tuple:
    if os.path.exists(path):
      extra = open(path+"var.txt", "r").readlines()[0].split(" ")
      return load_model(path+"model.h5"), list(map(int, extra[:-1]))+[extra[-1]]
    
