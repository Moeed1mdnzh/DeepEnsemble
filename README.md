# DeepEnsemble
I brought the VotingClassifier from sklearn to deep networks such as **CNNs**, **ANNs**, **RNNs** and etc to provide the capability of training deep networks by ensembling.The
repo consists 2 separate files <br /> *singular_ensemble.py* <br /> *customized_ensemble.py* <br />
## Main-steps
#### step 1
Clone the repo using the below command<br />
`git clone https://github.com/Moeed1mdnzh/DeepEnsemble.git`<br />
or just simply download the zipped repo
#### step 2
Install the required packages
```python
pip install -r requirements.txt 
``` 
### singular_ensemble.py
The point of this file is to build ensembles of only a unique architecture.<br />
Let's find out how you can use it in your very own models.
#### step 1
Put the file in same directory as your own files and import the ensembling-object in your main file.
```python
from singular_ensemble import SingularEnsemble
``` 
#### step 2
Build the ensembling-object based on your own conditions
```python
base = SingularEnsemble(layers, classes, n_estimators, voting = "hard", verbose = 2, save_to = False)
``` 
1-*layers* : **Must be a list which containes all the layers of your model**<br />
2-*classes* : **The number of your data classes**<br />
3-*n_estimators* : **The number of the estimators to be created for your model**<br />
4-*voting* : **The type of voting for making predictions(The same argument in sklearn.ensemble.VotingClassifer)**<br />
5-*verbose* : **Clears the screen after the given number of fits**<br />
6-*save_to* : **Supposed to be the path of saving the model.If given a string,the model will be saved there otherwise leave it**<br />
#### step 3
Compile the model by passing a dictionary as the parameters.For instance
```python
base.compile({"loss":"binary_crossentropy", "optimizer":"adam", "metrics":["accuracy"]})
``` 
Fit the model in the same way.
```python
base.fit({"x":X_train, "y":y_train, "batch_size":32, "epochs":10, "validation_data":(X_test, y_test)})
``` 
After fitting the model you should see two new files named `model.h5` and `var.txt` if `save_to` in the ensembling-object is set to a path.<br />
`model.h5` is the saved model and `var.txt` contains some important variables so don't touch them.
#### step 4
Connect all models into one and save it(if `save_to` is set to a path) by typing
```python
base.aggregate()
``` 
#### step 5
Load And Predict
```python
from singular_ensemble import SingularEnsemble as base
model, extra = base.load(PATH)
prediction = base.predict(model, sample, extra)
```
#### step 6(optional)
Evaluate the model
```python
preformance = base.evaluate(model, X_test, y_test, extra, batch_size)
```
### customized_ensemble.py
The point of this file is to build ensembles of several individual architecture.<br />
Let's also see how this one works.How exciting!
#### step 1
Put the file in same directory as your own files and import the ensembling-object in your main file.
```python
from customized_ensemble import CustomizedEnsemble
``` 
#### step 2
Build the ensembling-object based on your own conditions
```python
base = CustomizedEnsemble(models : list, voting : str = "hard", verbose : int = 2, save_to = False)
``` 
1-*models* : **Must be a list which containes all the various models**<br />
2-*voting* : **The type of voting for making predictions(The same argument in sklearn.ensemble.VotingClassifer)**<br />
3-*verbose* : **Cleara the screen after the given number of fits**<br />
4-*save_to* : **Supposed to be the path of saving the model.If given a string,the model will be saved there otherwise leave it**<br />

#### step 3
Compile the model by passing a list which contains all the parameters for each model as a dictionary.For instance
```python
base.compile([{"loss":"binary_crossentropy", "optimizer":"adam", "metrics":["accuracy"]},
              {"loss":"binary_crossentropy", "optimizer":"rmsprop", "metrics":["accuracy"]},
              {"loss":"binary_crossentropy", "optimizer":"sgd", "metrics":["accuracy"]}])
``` 
Fit the model in the same way.
```python
base.fit([{"x":X_train, "y":y_train, "batch_size":32, "epochs":10, "validation_data":(X_test, y_test)},
          {"x":X_train, "y":y_train, "batch_size":32, "epochs":10, "validation_data":(X_test, y_test)},
          {"x":X_train, "y":y_train, "batch_size":32, "epochs":5, "validation_data":(X_test, y_test)}])
``` 
After fitting the model you should see two new files named `model.h5` and `var.txt` if `save_to` in the ensembling-object is set to a path.<br />
`model.h5` is the saved model and `var.txt` contains some important variables so don't touch them.
#### step 4
Connect all models into one and save it(if `save_to` is set to a path) by typing
```python
base.aggregate()
``` 
#### step 5
Load And Predict except we have to do an extra thing which is to decompose the model back to separated models.
```python
from customized_ensemble import CustomizedEnsemble as base
model, extra = base.load(PATH)
models = base.decompose(model, extra)
prediction = base.predict(models, sample, extra)
```
#### step 6(optional)
Evaluate the model
```python
performance = base.evaluate(models, X_test, y_test, extra, batch_size)
```
Hopefully you succeeded at creating your own ensembling model :)
