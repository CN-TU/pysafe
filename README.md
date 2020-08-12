# pysafe
Contact: Fares Meghdouri - fares.meghdouri@tuwien.ac.at
Python implementation of the per-SAmple Feature Elimination algorithm

## Installation

pyodm can be installed using pip by running
```pip install git+https://github.com/CN-TU/pysafe```

## Usage

### Create a model
```python
from pysafe import SAFE

# create a new model with default parameters
safe_model = SAFE('forward_selection', random_state=2020)
```

### Scan a Keras model
```python
import numpy
#import pandas

# read the data
X = np.load('my_data.npy')
y = np.load('my_labels.npy')
#or
#X = pandas.read_csv('my_data.csv')
#X = pandas.read_csv('my_datalabels.csv')

# scan the model for adversarial samples
safe_model.scan(keras_model, X, y, aim='worst')

# scan the model for improving accuracy
#safe_model.scan(keras_model, X, y, aim='better')

# get the robustness of each feature
print(safe_model.get_robustness())

# get the subliminal dataset data
print(safe_model.X)

# get the subliminal dataset labels
print(safe_model.y_worst)
print(safe_model.y_better)
```

### Learn Combinations With a SAFE Model
```python
# if you want to use knn
safe_model.learn(algorithm = 'knn')

# if you want to use ann with predefined model
#safe_model.learn(algorithm = 'ann')

# if you want to pass your pre-trained model
#safe_model.learn(algorithm = 'ann', learner=my_model)

# if you want to pass your empty model
#safe_model.learn(algorithm = 'ann', learner=my_model, train=True)
```

### Robustness of Features in a Dataset
```python
safe_model.get_robustness(data=my_dataset)
```

### Features to Select in a Test Set
```python
# to get the combinations
safe_model.get_selection(data=my_dataset)

# to apply the combinations on a dataset (without typing the previous command)
safe_model.clean_data(data=my_dataset)
```

### Compare Accuracy Before and After SAFE
```python
safe_model.get_accuracy(data=my_data, labels=my_labels)
```

### Behaviour of a Sample Before and After SAFE
```python
safe_model.get_behaviour(data=my_data_sample)
# this will yield the prediction before and after SAFE
```

### Get Candidates Adversarial Saamples
```python
# get all candidates in a dataset that can chnage their prediction by a value higher than a threshold (0.9)
safe_model.get_candidates(data=my_dataset, threshold=0.9)
```
