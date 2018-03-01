import numpy as np
from sklearn.metrics import f1_score
from utils import load_data, train_test_split

X, Y = load_data("iris.csv")

X_tr, Y_tr, X_t, Y_t = train_test_split(X, Y, .7)

class KNN:
    def __init__(self):
        """
        Your initialization procedure if required
        """
        pass

    def fit(self, X, Y):
        """
        KNN algorithm in the simples implementation can work only with
        continuous features

        X: training data, numpy array of shape (n,m)
        Y: training labels, numpy array of shape (n,1)
        """

        # Hint: make sure the data passed as input are of the type float
        # Hint: make sure to create copies of training data, not copies of
        #       references

        self.X_tr = X
        self.Y_tr = Y.ravel()

    def predict(self, X, nn=5):
        distance = 0
        distances = []
        prediction = []
        for i in range(X_t.shape[0]):
            for j in range(X.shape[0]):
                distance += np.sqrt(np.sum(np.square(np.array(X_t[i]) - np.array(X[j]))))
                distances.append(distance)

            indexes = np.argsort(distances)
            top_nn = indexes[:nn]
            closest_elts = self.Y_tr[top_nn]
            dict = {}
            for cl in np.unique(closest_elts):
                dict[cl] = 0
            for idx in top_nn:
                dict[self.Y_tr[idx]] += 1 / (distances[idx] + 1)
            max_key = max(dict, key=lambda k: dict[k])
            print(max_key)
            prediction.append(max_key)
        return np.array(prediction)


# Task:
# 1. Implement function fit in the class KNN
# 2. Implement function predict in the class KNN, where neighbours are weighted
#     according to uniform weights
# 3. Test your algorithm on iris dataset according to
#     f1_score (expected: 0.93)
# 4. Test your algorithm on mnist_small dataset according to
#     f1_score (expected: 0.7)
# 5. Test your algorithm on mnist_large dataset according to
#     f1_score (expected: 0.86)
# 6. Implement function predict in the class KNN, where neighbours are weighted
#     according to their distance to the query instance


np.random.seed(1)

c = KNN()
c.fit(X_tr, Y_tr)
label_p = c.predict(X_t)
f1 = f1_score(Y_t[:,0], label_p, average='micro')
print("Test score %.2f" % f1 )


