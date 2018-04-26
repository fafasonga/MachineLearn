import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import f1_score
from utils import load_data, train_test_split

# Start session
sess = tf.Session()

x_vals, y_vals = load_data("mnist_small.csv")
y_vals = np.where(y_vals < 5, 0, 1)

    # Splitting the dataset in Training Set and Test set
x_train, y_train, x_test, y_test = train_test_split(x_vals, y_vals, 0.8)

    # Network Parameters
learning_rate = 0.01
num_classes = y_test.shape                  # MNIST total classes (0-9 digits)
num_steps = 500                             # Total Step Size
x_size, num_feature = x_vals.shape          # MNIST data input (img shape: 28*28)
n_hidden_1 = 100                            # 1st layer number of neurons
n_hidden_2 = 100                            # 2nd layer number of neurons

    # tf Graph input
X = tf.placeholder(tf.float32, shape=[None, num_feature], name="x-input")
Y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y-input")
Y = tf.cast(Y, tf.float32)



    # Create model
def neural_net(x):
    # Hidden fully connected layer with 100 neurons
    layer_1 = tf.layers.dense(inputs=x, units=n_hidden_1, activation=tf.nn.sigmoid)
    # Hidden fully connected layer with 100 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_1, activation=tf.nn.sigmoid)
    # Output fully connected layer with a neuron for each class
    layer_out = tf.layers.dense(inputs=layer_2, units=1)
    return layer_out

logits = neural_net(X)

    # Cross Entropy Calculation
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y, name='Cross_Entropy')

losse = tf.reduce_mean(cross_entropy, name='Loss')

    # Prediction
predict = tf.round(tf.nn.sigmoid(logits))


    # Optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(losse)

    # finally setup the initialisation operator
init_op = tf.global_variables_initializer()
sess.run(init_op)

loss_out = []
for i in range(num_steps):
    sess.run(opt, feed_dict={X:x_train, Y:y_train})
    valu = sess.run(losse, feed_dict={X:x_train, Y:y_train})
    print(valu)
    loss_out.append(valu)

    if i == 0 or (i + 1) % 100 == 0:
        print(i + 1, valu)
