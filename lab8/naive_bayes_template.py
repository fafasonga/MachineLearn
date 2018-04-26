import numpy as np
import math
from pandas import get_dummies, DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, KFold
from utils import load_data

# function to calculate mean
def estimate_mean(values):
    return sum(values) / float(len(values))

# function to calculate Standard Deviation
def estimate_stdev(values):
    avg = estimate_mean(values)
    variance = sum([pow(x - avg, 2) for x in values]) / float(len(values) - 1)
    print(variance)
    return math.sqrt(variance)


class NaiveBayes:
    def __init__(self):
        """
        Your initialization procedure if required
        """
        self.classe_prob = {}
        self.cond_prob = {}
        self.classe = {}

        pass

    def fit(self, X, Y):
        """
        This method calculates class probabilities and conditional probabilities to be used for prediction

        Both numerical and categorical features are accepted.
        Conditional probability of numerical features is calculated based on Probability Density Function
        (assuming normal distribution)

        :param X: training data, numpy array of shape (n,m)
        :param Y: training labels, numpy array of shape (n,1)
        """
        # TODO START YOUR CODE HERE

        self.classe = np.unique(Y)
        class_size = Y.size
        self.class_val = {}

        i = 0
        for c in self.classe:
            self.class_val[c] = i
            i += 1
            # Posterior Class Probability P(c) = P(c|x)
            self.classe_prob[c] = len([y for y in Y if y == c]) / class_size

        self.cond_prob = {}

        # Prob(feature = value | class)
        for f in range(X.shape[1]):
            data_sample = np.unique(X[:, f])
            self.cond_prob[f] = {}

            for c in self.classe:
                class_data = len([y for y in Y if y == c])
                self.cond_prob[f][c] = {}
                for v in data_sample:
                    self.cond_prob[f][c][v] = len(
                        [i for i in range(X.shape[0]) if X[i][f] == v and Y[i] == c]) / class_data
                print(f, c, self.cond_prob[f][c])

        # END YOUR CODE HERE

    @staticmethod
    def estimate_mean_and_stdev(self, values):
        """
        Estimates parameters of normal distribution - empirical mean and standard deviation
        :param values: attribute sample values
        :return: mean, stdev
        """
        # TODO START YOUR CODE HERE

        mean = estimate_mean(values)
        stdev = estimate_stdev(values)
        class_probs = {}
        for i in range(len(values)):
            for c in self.classe:
                class_probs[c] *= self.calc_probability(X, mean, stdev)

        return class_probs

        # END YOUR CODE HERE


    @staticmethod
    def calc_probability(val, mean, stdev):
        """
        Estimates probability of encountering a point (val) given parameters of normal distribution
        based on probability density function
        :param val: point
        :param mean: mean value
        :param stdev: standard deviation
        :return: relative likelihood of a point
        """
        # TODO START YOUR CODE HERE

        exponent = math.exp(-(math.pow(val - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

        # END YOUR CODE HERE

    def predict(self, X):
        """
        Predict class labels for given input. Refer to lecture slides for corresponding formula
        :param X: test data, numpy array of shape (n,m)
        :return: numpy array of predictions
        """
        # TODO START YOUR CODE HERE

        predictor = []
        data_prob = {}

        for i in range(X.shape[0]):
            data_c_prob = {}
            for c in self.classe:
                data_c_prob = self.classe_prob[c]
                for f in range(X.shape[1]):
                    valu = X[i][f]

                    if valu in self.cond_prob[f][c]:
                        data_c_prob *= self.cond_prob[f][c][valu]

                    else:
                        diction_val = self.cond_prob[f][c].values()

                        # Zero frequency for every class value attribute combination
                        data_weigth = sum(diction_val) / (len(diction_val) + 1)
                        data_c_prob *= data_weigth

                data_prob[c] = data_c_prob
                print("Prob for each attribute", data_prob)

            # Returning the Maximum element
            max_value = max(data_prob, key=lambda h: data_prob[h])
            predictor.append(max_value)

        return np.array(predictor)

        # END YOUR CODE HERE

    def get_params(self, deep = False):
        return {}


X, Y = load_data("crx.data.csv")
# indexes of numerical attributes
numerical_attrs = [1, 2, 7, 10, 13, 14]
X[:, numerical_attrs] = X[:, numerical_attrs].astype(float)

# categorical features only. Use this to test your initial implementation
X_cat = np.delete(X, numerical_attrs, 1)

model = NaiveBayes()
model.fit(X, Y)
model.predict(X)

scores = cross_val_score(NaiveBayes(), X_cat, Y, cv=KFold(n_splits=15, shuffle=True), scoring='accuracy')
print("Categorical Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# use this as a benchmark. Your algorithm (on categorical features) should reach the same accuracy
X_dummy = DataFrame.as_matrix(get_dummies(DataFrame(X_cat)))
scores = cross_val_score(MultinomialNB(), X_dummy, Y.ravel(), cv=KFold(n_splits=15, shuffle=True), scoring='accuracy')
print("Categorical Accuracy of Standard NB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# all (mixed) features. Use this to test your final implementation
scores = cross_val_score(NaiveBayes(), X, Y.ravel(), cv=KFold(n_splits=15, shuffle=True), scoring='accuracy')
print("Overall Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# write your thoughts here (if any)

# c represent ith class of classes{1,2,3 ..n}
# Posterior Probability simply means, “given the feature vector xi what is the probability of sample i belonging to class c?”
# The onjective is to maximize the posterior probability given the training data to formulate the decision rule of new data.
# In Zero Frequency Smooting we took sum of number of times x appears in sample class c / total count of all features in class c
# The Categorical Accuracy and Categorical Accuracy of Standard NB are same and have 0.86
# The Averall Accuracy of all the feautures, to say Categorical and data sample are 0.67
