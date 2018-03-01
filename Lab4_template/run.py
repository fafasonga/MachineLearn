from pandas import read_csv
import numpy as np


def load_data(path_to_csv, has_header=True):
    """
    Loads a csv file, the last column is assumed to be the output label
    All values are interpreted as strings, empty cells interpreted as empty
    strings

    returns: X - numpy array of size (n,m) of input features
             Y - numpy array of output features
    """
    if has_header:
        data = read_csv(path_to_csv, header='infer', dtype=str)
    else:
        data = read_csv(path_to_csv, header=None, dtype=str)
    data.fillna('', inplace=True)
    data = data.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


class DTree:
    """
    Simple decision tree classifier for a training data with categorical
    features
    """
    _model = None

    def fit(self, X, Y):
        self._model = create_branches({'attr_id': -1,
                                       'branches': dict(),
                                       'decision': None}, X, Y)

    def predict(self, X):

        if X.ndim == 1:
            return traverse(self._model, sample)
        elif X.ndim == 2:
            Y_pred = []
            for sample in X:
                Y_pred.append(traverse(self._model, sample))
            return np.array(Y_pred)
        else:
            print("Dimensions error")

    def prune(self, X_val, Y_val, metric):
        """
        Implement pruning to improve generalization
        """

        # Start pruning from the root node
        while True:  
            curr_node = self._model
            acc_node = metric(Y_val, self.predict(X_val))
            # Traversing in column
            while True:
                next_node = False
                for attr_val in curr_node['branches']:
                    if curr_node['branches'][attr_val]['attr_id'] != -1:
                        curr_node = curr_node['branches'][attr_val]
                        next_node = True
                        break
                if not next_node:
                    break
            print("Pruning node: ", curr_node)
            save_node = curr_node.copy()
            curr_node['attr_id'] = -1
            curr_node['branches'] = dict()
            acc_child = metric(Y_val, self.predict(X_val))
            if acc_child < acc_node:
                curr_node = save_node.copy()
                print("Finished Pruning")
                return
            else:
                print("Continue Pruning: ", acc_child)



def elem_to_freq(values):
    """
    input: numpy array
    returns: The counts of unique elements, unique elements are not returned
    """
    # hint: check numpy documentation for how to count unique values
    freqs = []
    for value in np.unique(values):
        freqs.append(len(values[values == value]))
    return np.array(freqs) / np.sum(freqs)


def entropy(elements):
    """
    Calculates entropy of a numpy array of instances
    input: numpy array
    returns: entropy of the input array based on the frequencies of observed
             elements
    """
    # hint: use elem_to_freq(arr)

    freqs = elem_to_freq(elements)
    freqs = freqs[freqs > 0]
    entropy = -np.sum(freqs * np.log2(freqs))
    return entropy


def information_gain(A, S):
    """
    input:
        A: the values of an attribute A for the set of training examples
        S: the target output class

    returns: information gain for classifying using the attribute A
    """
    # hint: use entropy(arr)

    gain = entropy(S)

    # We calculate a weighted average of the entropy
    for value in np.unique(A):
        v = np.sum(A == value) / len(A)
        gain -= v * entropy(S[A == value])
        print("Gain is: ", gain)

        return gain


def choose_best_attribute(X, Y):
    """
    input:
        X: numpy array of size (n,m) containing training examples
        Y: numpy array of size (n,) containing target class

    returns: the index of the attribute that results in maximum information
             gain. If maximum information gain is less that eps, returns -1
    """

    eps = 1e-10

    max_gain = eps
    best_attr = -1
    for value in range(X.shape[1]):
        gain = information_gain(X[:, value], Y)
        if gain > max_gain:
            max_gain = gain
            best_attr = value

    return best_attr


def most_common_class(Y):
    """
    input: target class values
    returns: the value of the most common class
    """
    max_count = 0
    max_val = None
    for value in np.unique(Y):
        if len(Y[Y == value]) > max_count:
            max_count = len(Y[Y == value])
            max_val = value
    return max_val


def create_branches(node, X, Y):
    """
    create branches in a decision tree recursively
    input:
        node: current node represented by a dictionary of format
                {'attr_id': -1,
                 'branches': dict(),
                 'decision': None},
              where attr_id: specifies the current attribute index for branching
                            -1 mean the node is leaf node
                    braches: is a dictionary of format {attr_val:node}
                    decision: contains either the best guess based on
                            most common class or an actual class label if the
                            current node is the leaf
        X: training examples
        Y: target class

    returns: input node with fields updated
    """
    # choose best attribute to branch
    attr_id = choose_best_attribute(X, Y)
    node['attr_id'] = attr_id
    # record the most common class
    node['decision'] = most_common_class(Y)

    if attr_id != -1:
        # find the set of unique values for the current attribute
        attr_vals = np.unique(X[:,attr_id])

        for a_val in attr_vals:
            # compute the boolean array for slicing the data for the next
            # branching iteration
            # hint: use logical operation on numpy array
            # for more information about slicing refer to numpy documentation
            sel = (X[:,attr_id] == a_val)
            # perform slicing
            X_branch = X[sel, :]
            Y_branch = Y[sel]
            node_template = {'attr_id': -1,
                             'branches': dict(),
                             'decision': None}
            # perform recursive call
            node['branches'][a_val] = create_branches(node_template, X_branch, Y_branch)
    return node


def traverse(model, sample):
    """
    recursively traverse decision tree
    input:
        model: trained decision tree
        sample: input sample to classify

    returns: class label
    """
    if model['attr_id'] == -1:
        decision = model['decision']
    else:
        attr_val = sample[model['attr_id']]
        if attr_val not in model['branches']:
            decision = model['decision']
        else:
            decision = traverse(model['branches'][attr_val], sample)
    return decision


def train_test_split(X, Y, fraction):
    """
    perform the split of the data into training and testing sets
    input:
        X: numpy array of size (n,m)
        Y: numpy array of size (n,)
        fraction: number between 0 and 1, specifies the size of the training
                data

    returns:
        X_train
        Y_train
        X_test
        Y_test
    """
    if fraction < 0 or fraction > 1:
        raise Exception("Fraction for split is not valid")

    whole_data = np.arange(0, len(Y))
    train_data = np.random.choice(whole_data, size= int(fraction * len(Y)), replace = False)
    test_data = np.array([i for i in whole_data if i not in train_data])
    return X[train_data], Y[train_data], X[test_data], Y[test_data]



def measure_error(Y_true, Y_pred):
    """
    returns an error measure of your choice
    """
    return np.sum(Y_true == Y_pred) / len(Y_true)


def recall(Y_true, Y_pred):
    """
    returns recall value
    """
    if "yes" in Y_true:
        true_label = "yes"
    elif 1 in Y_true:
        true_label = 1
    else:
        true_label = Y_true[0]
    tp = np.sum((Y_pred == Y_true) & (Y_true == true_label))
    return tp / np.sum(Y_true == true_label)


# 1.  test your implementation on data_1.csv
#     refer to lecture slides to verify the correctness
# 2.  test your implementation on mushrooms_modified.csv
# 3.  test your implementation on titanic_modified.csv


X,Y = load_data("data_1.csv")
# X,Y = load_data("mushrooms_modified.csv")
# X,Y = load_data("titanic_modified.csv")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, .8)

d_tree = DTree()
d_tree.fit(X_train,Y_train)
Y_pred = d_tree.predict(X_test)
print("Correctly classified: %.2f%%" % (measure_error(Y_test, Y_pred) * 100))
print("Recall %.4f" % recall(Y_test, Y_pred))

