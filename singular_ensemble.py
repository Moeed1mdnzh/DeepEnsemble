import os
import numpy as np
from scipy.stats import mode
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import  clone_model

#  Only trainable for a specific architecture
class SingularEnsemble:
    def __init__(self, layers : list, classes : int, n_estimators : int = 50, voting : str = "hard", sampling : bool = False, bootstrap : bool = True, verbose : int = 1, save_to = False):
        self.org = layers 
        self.n_estimators = n_estimators
        self.voting = voting
        self.models = []
        self.sampling = sampling
        self.bootstrap = bootstrap 
        self.verbose = verbose
        self.mainModel = None
        self.classes = classes
        if save_to:
            self.path = save_to

#-----------------------------------------------------------------------------------------------------------------------------------------|
                                                                                                                                          
    #  Train the model by giving the parameters as a dictionary
    def fit(self, params : dict) -> None:
        if self.sampling:
            new_data = self.sample(params["x"], params["y"])
        for i in range(1, self.n_estimators+1):
            if i % self.verbose == 0 or i == 1:
                os.system("cls")
            print(f"Fitting model number {i}. . .")
            if self.sampling:
                subset = next(new_data)
                params["x"] = subset["x"]
                params["y"] = subset["y"]
            model = self.create(self.org, i)
            self.compile_model(model)
            model.fit(**params)
            self.models.append(model)

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Compile parameter unpacker
    def compile_model(self, model):
        return model.compile(**self.compParams)

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Compile the model by the parameters as a dictionary
    def compile(self, params : dict) -> None:
        self.compParams = params

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Perform bagging and pasting
    def sample(self, X, y):
        data = np.concatenate((X, y), axis=1)
        np.random.shuffle(data)
        num_of_subsets = int(len(data) * 0.63)
        for _ in range(self.n_estimators):
            indices = np.random.choice(np.arange(X.shape[0]), num_of_subsets, replace = self.bootstrap)
            new = data[indices]
            yield {"x": new[:, :X.shape[1]], "y": new[:, X.shape[1]:]}

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Changes layer names to avoid repeated layer names
    def create(self, layers : list, pos : int):
        model = Sequential()
        for i, layer in enumerate(layers):
            name = layer._name[:layer._name.index("_")+1]
            layer._name = f"{name}{pos}-{i}"
            model.add(layer)
        return clone_model(model)

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Connect all the individual models into one
    def aggregate(self) -> None:
        details = np.array([[model.inputs, model.output] for model in self.models])
        inputs, outputs = details[:, 0], details[:, 1]
        conn = Concatenate()(list(outputs))
        self.mainModel = Model(list(inputs), conn)
        if os.path.exists(self.path):
            with open(self.path+"var.txt", "w") as f:
                f.write(f"{self.n_estimators} {self.voting} {self.classes}")
                f.close()
            self.mainModel.save(self.path+"model.h5")
  
#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Make predictions
    @staticmethod
    def predict(model, sample : np.ndarray, extra : list) -> int:
        predictions = []
        preds = model.predict([sample for _ in range(extra[0])])
        preds = np.hsplit(preds, extra[0])
        pred = None
        if extra[1] == "hard":
            for pred in preds:
                predictions.append(np.argmax(pred))
            pred = mode(np.array(predictions)).mode[0]
        elif extra[1] == "soft":
            preds = np.sum(np.array(preds), axis=0)
            pred = np.argmax(preds)
        return pred

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Model evaluation
    @staticmethod
    def evaluate(model, X_test : np.ndarray, y_test : np.ndarray, extra : list, batch_size : int) -> list:
        losses, accs = [], []
        to_bin = False
        if y_test.ndim > 1:
            to_bin = True
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            preds = np.array([model.predict([batch[i].reshape(-1, *X_test.shape[1:]) for j in range(extra[0])]) for i in range(len(batch))])
            preds = preds.reshape(len(preds), extra[0], extra[2])
            preds = np.argmax(np.sum(preds, axis=1), axis=1)
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

#-----------------------------------------------------------------------------------------------------------------------------------------|

    #  Load the model and some extra values
    @staticmethod
    def load(path : str) -> tuple:
        if os.path.exists(path):
            extra = open(path+"var.txt", "r").readlines()[0].split(" ")
            return load_model(path+"model.h5"), [int(extra[0]), extra[1], int(extra[2])]
