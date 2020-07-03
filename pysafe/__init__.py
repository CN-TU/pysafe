import itertools
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.model_selection import train_test_split
from evolutionary_search import maximize
from tqdm import tqdm

class SAFE():
    """ per SAmple Feature Elimination """

    def __init__(self, mode='all', factor=1, n_points=None, random_state=None):

        self.n_features = None
        self.mode = mode
        self.combinations = None
        self.tree = None
        self.X_train_fs = None
        self.y_train_fs = None
        self.model = None
        self.n_points = n_points
        self.random_state = random_state
        self.factor = factor

    def fit(self, model, X_train, y_train):

        if self.n_points is not None:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=self.n_points/X_train.shape[0], random_state=self.random_state, stratify=y_train)

        self.X_train_fs = np.zeros(X_train.shape)
        self.y_train_fs = np.zeros(X_train.shape)
        self.n_features = self.X_train_fs.shape[1]
        self._generate_combination(self.mode)
        self.model = model

        
        if self.mode == 'genetic':
            param_grid = {'combination': self.combinations}
            for i,j in enumerate(tqdm(X_train.values)):
                args = {'data': j, 'label':y_train.values[i]}
                best_params, _, _, _, _ = maximize(self._combination_search, param_grid, args, verbose=False)
                self.X_train_fs[i,:] = j
                self.y_train_fs[i,:] = best_params['combination']
        else:
            for i,j in enumerate(tqdm(X_train.values)):
                candidates = []
                for p in self.combinations:
                    sample = np.multiply(p, j)
                    loss = sum(abs(model.predict(np.array([sample,]))[0] - y_train.values[i]))
                    candidates.append(loss)
                self.X_train_fs[i,:] = j
                self.y_train_fs[i,:] = self.combinations[np.array(candidates).argmin()]

    def get_selection(self, test_data):
        self.tree = KDTree(self.X_train_fs)
        _, ind = self.tree.query(test_data, k=1)
        return self.y_train_fs[ind.flatten()]

    def get_importance(self, data=None):
        if data is not None:
            y = self.get_selection(data)
        else:
            y = self.y_train_fs

        try:
            return np.mean(y, 0)/np.sum(np.mean(y, 0))*100
        except:
            raise "SAFE model not trained! Use the fit() mnethod for training."

    def _generate_combination(self, mode):
        if mode == 'all' or mode == 'genetic':
            self.combinations =  list(itertools.product([0, 1], repeat=self.n_features))
        if mode == 'one-by-one':
            self.combinations = np.ones((self.n_features+1, self.n_features))
            for i in range(self.n_features):
                self.combinations[i,i] = 0
        if mode == 'random':
            pass

        # TODO: sample based on self.factor

    def clean_data(self, data):
        y = self.get_selection(data)
        return np.multiply(X_test, y)

    def _combination_search(self, combination, data, label):
        sample = np.multiply(data, combination)
        return 1/(sum(abs(self.model.predict(np.array([sample,]))[0] - label)))

    def get_accuracy(self, data, labels):

        original = data
        cleaned = self.clean_data(data)

        original_predictions = self.model.predict_classes(original)
        cleaned_predictions = self.model.predict_classes(cleaned)
        
        try:
            print('Accuracy before: {}'.format(accuracy_score(np.argmax(labels, 1), original_predictions)))
            print('Accuracy after: {}'.format(accuracy_score(np.argmax(y_test, 1), cleaned_predictions)))
        except:
            print('Accuracy before: {}'.format(accuracy_score(labels, original_predictions)))
            print('Accuracy after: {}'.format(accuracy_score(y_test, cleaned_predictions)))
