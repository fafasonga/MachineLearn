import tensorflow as tf
from utils import load_data
from sklearn.model_selection import train_test_split

data, target = load_data("data/weatherHistory.csv")
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3)

num_features = X_train.shape[1]
w_shape = (1, num_features)
b_shape = (1, 1)   # 0
epochs = 100

X = tf.placeholder(shape=(None, num_features), dtype=tf.float32, name="input")
Y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="target")

W = tf.get_variable(shape=w_shape, dtype=tf.float32, name="weights", initializer=None)
b = tf.get_variable(shape=b_shape, dtype=tf.float32, name="bias", initializer=None)

Y_hat = tf.transpose(tf.multiply(W, tf.transpose(X)) + b)
loss = tf.reduce_mean(tf.square(Y - Y_hat, name="loss"))
print("Loss is : ", loss)

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
# print("Optimation is :", optimizer)

opt = optimizer.minimize(loss)
print("The Minimized optimization is :", opt)

# Evaluating all Variables

initializer = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initializer)

    # Evaluation of session objects
    for i in range(epochs):
        # Perform Single Optimization step
        _, loss_train = sess.run([opt, loss], {X: X_train, Y: Y_train})
        loss_test = sess.run(loss, {X: X_test, Y: Y_test})
        print("\rIteration: %d "
              "Train loss: %0.2f "
              "Test loss: %0.2f " % (i, loss_train, loss_test), end="")
        print("\nWeights: ", sess.run(W))

# logits = tf.transpose(tf.multiply(W, tf.transpose(X)) + b)
# print("Logits is : ", logits)
#
# logi_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
# print("Logistic Regression is : ", logi_loss)
