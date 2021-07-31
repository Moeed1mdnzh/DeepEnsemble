# DeepEnsemble
I brought the VotingClassifier from sklearn to deep networks such as **CNNs**, **ANNs** and etc to provide the capability of training deep networks by ensembling.The
repo consists 2 separate files <br /> *singular_ensemble* <br /> *customized_ensemble* <br />
## Main-steps
#### step-1
Clone the repo using the below command<br />
`git clone https://github.com/Moeed1mdnzh/DeepEnsemble.git`<br />
or just simply download the zipped repo
#### step-2
Install the required packages
```python
pip install -r requirements.txt 
``` 
### singular_ensemble
The point of this file is to build ensembles of only a unique architecture.<br />
Let's find out how you can use it in your very own models.
#### step-1
Put the file in same directory as your own files and import the file and import the ensembling-object in your main file.
```python
from singular_ensemble import SingularEnsemble
``` 
#### step-2
Build the ensembling-object based on your own conditions
```python
base = SingularEnsemble(layers, classes, n_estimators, voting = "hard", verbose = 2, save_to = False)
``` 
1-*layers* : **Must be a list which containes all the layers of your model**
2-*classes* : **The number of your data classes**
3-*n_estimators* : **The number of the estimators to be created for your model**
4-*voting* : **The type of voting for making predictions(The same argument in sklearn.ensemble.VotingClassifer)**
4-*verbose* : **Defines clearing the screen after how many fits**
5-*save_to* : **Supposed to be the path of saving the model.If given a string,the model will be saved there otherwise leave it**
#### step-3
Compile the model by passing a dictionary as the parameters.For instance
```python
base.compile({"loss":"binary_crossentropy", "optimizer":"adam", "metrics":["accuracy"]})
``` 
Fit the model in the same way.
```python
base.fit({"x":X_train, "y":y_train, "batch_size":32, "epochs":10, "validation_data":(X_test, y_test)})
``` 
#### step-4
Connect all models into one and save it(if `save_to` is set to a path) by typing
```python
base.aggregate()
``` 
#### step-5
Load And Predict
```python
from singular_ensemble import SingularEnsemble
model, extra = base.load(PATH)
prediction = base.predict(model, sample, extra)
```
#### step-6(optional)
Evaluate the model
```python
evaluate(model, X_test, y_test, extra, batch_size)
```

evaluate(model, X_test : np.ndarray, y_test : np.ndarray, extra : list, batch_size : int, loss_func = None)

``` 
