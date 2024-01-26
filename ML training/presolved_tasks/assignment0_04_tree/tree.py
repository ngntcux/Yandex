import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    probabilities = np.mean(y, axis=0)
    return float(-np.sum(probabilities * np.log(probabilities + EPS)))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    if len(y) == 0:
        print('nan!')
    probabilities = np.mean(y, axis=0)
    return float(1 - np.sum(probabilities**2))
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    return float(np.var(y))

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    return float(np.mean(np.abs(y - np.median(y))))

def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index=None, threshold=None, targets=None, left_child=None, right_child=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.targets = targets
        self.left_child = left_child
        self.right_child = right_child
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        mask = X_subset[:, feature_index] < threshold
        X_left, y_left = X_subset[mask], y_subset[mask]
        X_right, y_right = X_subset[~mask], y_subset[~mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        mask = X_subset[:, feature_index] < threshold
        y_left = y_subset[mask]
        y_right = y_subset[~mask]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """

        if len(X_subset) < self.min_samples_split:
            return None, None
        criterion, _ = self.all_criterions[self.criterion_name]
        best_feature_index, best_threshold = None, None
        min_value = float('inf')
        for feature_index in range(X_subset.shape[1]):
            thresholds = np.unique(X_subset[:, feature_index])
            for threshold in thresholds:
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                value = len(y_left) * criterion(y_left) + len(y_right) * criterion(y_right)
                if value < min_value:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    min_value = value
        feature_index, threshold = best_feature_index, float(best_threshold)
        
        return feature_index, threshold
    
    def make_tree(self, X_subset, y_subset, depth=1):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        _, classification = self.all_criterions[self.criterion_name]

        if depth >= self.max_depth or len(X_subset) < self.min_samples_split:
            return Node(targets=y_subset)
            
        if classification and np.any(np.all(y_subset == 1, axis=0)):
            return Node(targets=y_subset)
        elif np.allclose(y_subset, y_subset[0]):
            return Node(targets=y_subset)
            
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        
        if len(X_left) == 0 or len(X_right) == 0:
            return Node(targets=y_subset)
            
        new_node = Node(feature_index=feature_index, threshold=threshold, left_child=self.make_tree(X_left, y_left, depth + 1), right_child=self.make_tree(X_right, y_right, depth + 1))
        
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        predictions = []
        _, classification = self.all_criterions[self.criterion_name]
        for sample in X:
            node = self.root
            while node.targets is None:
                if sample[node.feature_index] < node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            predictions.append(np.argmax(node.targets.sum(axis=0)) if classification else node.targets.mean())
        y_predicted = np.array(predictions).reshape(-1, 1)
                
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        probabilities = []
        for sample in X:
            node = self.root
            while node.targets is None:
                if sample[node.feature_index] < node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            probabilities.append(node.targets.mean(axis=0))
        y_predicted_probs = np.array(probabilities)
        
        return y_predicted_probs
