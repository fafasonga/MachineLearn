import tensorflow as tf
from utils import load_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data, target = load_data("data/weatherHistory.csv")
data, target = load_data("data/candy.csv")

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=.3)
num_features = X_train.shape[1]
w_shape = (1, num_features)
b_shape = (1, 1)
X = tf.placeholder(shape=(None, num_features), dtype=tf.float32, name="input")
Y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

W = tf.get_variable(shape=w_shape, dtype=tf.float32, name="weights", initializer=None)
b = tf.get_variable(shape=b_shape, dtype=tf.float32, name="bias", initializer=None)

Y_hat = tf.transpose(tf.matmul(W, tf.transpose(X)) + b)
loss_linear = tf.reduce_mean(tf.square(Y - Y_hat, name="loss"))

logi_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat))

optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
opt = optimizer.minimize(loss_linear)

opti = optimizer.minimize(logi_loss)
initializer = tf.global_variables_initializer()
epochs = 100

with tf.Session() as sess:
    sess.run(initializer)
    for i in range(epochs):
        _, loss_train = sess.run([opt, loss_linear], {X: X_train, Y: Y_train})
        loss_test = sess.run(loss_linear, {X: X_test, Y: Y_test})
        print("\rIteration: %d "
        "Train loss, Linear Regression: %.2f " "Test loss, Linear Regression: %.2f" %
        (i, loss_train, loss_test), end="")

        _, logi_loss_train = sess.run([opti, logi_loss], {X: X_train, Y: Y_train})
        logi_loss_test = sess.run(logi_loss, {X: X_test, Y: Y_test})
        print("\rIteration: %d "
              "Train loss, Logistic Regression: %.2f " "Test loss, Logistic Regression: %.2f" %
              (i, logi_loss_train, logi_loss_test), end="")


    print("\nWeights: ", sess.run(W))


    # plt.plot(X_test, Y_test, 'bo', label='Testing data')
    # plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()