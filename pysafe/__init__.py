import itertools
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.model_selection import train_test_split
from evolutionary_search import maximize
from tqdm import tqdm

class SAFE():
    """ per SAmple Feature Elimination """

    def __init__(self, combinations_mode='all', n_combinations=10, n_points=None, random_state=None):

        self.n_features = None
        self.combinations_mode = combinations_mode
        self.algorithm = None
        self.combinations = None
        self.tree = None
        self.X = None
        self.y_better = None
        self.y_worst = None
        self.model = None
        self.n_points = n_points
        self.random_state = random_state
        self.n_combinations = n_combinations
        self.tree = None
        self.learner = None

    def fit(self, model, X, y):

        [rows, self.n_features] = X.shape

        if self.n_points is not None:
            X, _, y, _ = train_test_split(X, y, train_size=self.n_points/rows, random_state=self.random_state, stratify=y)
        
        self.X = np.zeros((rows, self.n_features))
        self.y_better = np.zeros((rows, self.n_features))
        self.y_worst = np.zeros((rows, self.n_features))
        self._generate_combination()
        self.model = model

        losses = np.empty((rows, len(self.combinations)))
        
        # TODO migrate this also
        if self.combinations_mode == 'genetic':
            param_grid = {'combination': self.combinations}
            for i,j in enumerate(tqdm(X)):
                args = {'data': j, 'label':y_train.values[i]}
                best_params, _, _, _, _ = maximize(self.__combination_search, param_grid, args, verbose=False)
                self.X_train_fs[i,:] = j
                self.y_train_fs[i,:] = best_params['combination']
        else:
            for index, p in enumerate(tqdm(self.combinations)):
                losses[:, index] = abs(self.model.predict(np.multiply(p, X)).flatten() - y)
        
            self.y_worst = self.combinations[losses.argmax(axis=1)]
            self.y_better = self.combinations[losses.argmin(axis=1)]

    def learn(self, algorithm='knn', aim='worst', learner=None, train=False):
        self.y = self.y_better if aim == 'better' else self.y_worst
        self.algorithm = algorithm
        self.learner = learner if learner else None
        if self.algorithm == 'knn':
            if not learner:
                self.learner = KDTree(self.X)
        if self.algorithm == 'ann':
            self._ann(train)
        return self.learner

    def get_selection(self, data):

        if self.algorithm == 'knn':
            _, ind = self.learner.query(data, k=1)
            return self.y[ind.flatten()]
        if self.algorithm == 'ann':
            return self.learner.predict(data).round()

    def get_importance(self, data=None):
        if data is not None:
            y = self.get_selection(data)
        else:
            y = self.y

        try:
            return np.mean(y, 0)/np.sum(np.mean(y, 0))*100
        except:
            raise "SAFE model not trained! Use the fit() mnethod for training."

    def _generate_combination(self):
        if self.combinations_mode == 'all' or self.combinations_mode == 'genetic':
            self.combinations =  list(itertools.product([0, 1], repeat=self.n_features))
        if self.combinations_mode == 'one-by-one':
            self.combinations = np.ones((self.n_features+1, self.n_features))
            for i in range(self.n_features):
                self.combinations[i,i] = 0
        if self.combinations_mode == 'random':
            self.combinations = list(itertools.product([0, 1], repeat=self.n_features))
            self.combinations =  random.sample(self.combinations, min(self.n_combinations, len(self.combinations)))

    def clean_data(self, data):
        return np.multiply(data, self.get_selection(data))

    def __combination_search(self, combination, data, label):
        sample = np.multiply(data, combination)
        return 1/(sum(abs(self.model.predict(np.array([sample,]))[0] - label)))

    def _ann(self, train):

        if not self.learner:
            self.learner = Sequential()
            self.learner.add(Dense(64, input_dim=self.n_features, activation='relu'))
            self.learner.add(Dense(64, activation='relu'))
            self.learner.add(Dense(64, activation='relu'))
            self.learner.add(Dense(self.n_features, activation='sigmoid'))
            self.learner.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.learner.fit(self.X, self.y, epochs=2, batch_size=16)
            return
        
        if train:
            self.learner.fit(self.X, self.y, epochs=128, batch_size=8)
        
    def _get_clean(self, data):
        return self.model.predict_classes(data), self.model.predict_classes(self.clean_data(data))
    
    def get_accuracy(self, data, labels):
        original_predictions, cleaned_predictions = self._get_clean(data)
        
        # multiclass
        try:
            print('Accuracy before: {}'.format(accuracy_score(np.argmax(labels, 1), original_predictions)))
            print('Accuracy after: {}'.format(accuracy_score(np.argmax(labels, 1), cleaned_predictions)))
        # binary
        except:
            print('Accuracy before: {}'.format(accuracy_score(labels, original_predictions)))
            print('Accuracy after: {}'.format(accuracy_score(labels, cleaned_predictions)))
        
    def behaviour(self, data):
        return self.model.predict(data), self.model.predict(self.clean_data(data))
    
    def get_candidates(self, data, threshold=0.9):
        original_predictions, cleaned_predictions = self.behaviour(data)
        displacement = original_predictions - cleaned_predictions
        return [j for (i,j) in zip(displacement,list(range(len(displacement)))) if i >= threshold] 
