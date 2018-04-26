from utils import load_data, get_current_time, create_dirs, \
    create_minibatches, write_to_tensorboard, \
    create_summary_and_projector, create_evaluation_tensor
import tensorflow as tf
import os

# this project uses tensorboard. You can launch tensorboard by executing
# "tensorboard --logdir=log" in your project folder

# Set parameters
learning_rate = 0.001
minibatch_size = 125
num_epochs = 20
latent_space_size = 30
log_dir = "log"
current_run = get_current_time()

# Create necessary directories
log_path, run_path = create_dirs(log_dir, current_run)

# Load MNIST data
imgs, lbls = load_data()
mbs = create_minibatches(imgs, lbls, minibatch_size)

# Prepare evaluation set
# this set is used to visualize embedding space and decoding results
evaluation_set = mbs[0]
evaluation_shape = (minibatch_size, latent_space_size)

def create_model(input_shape):
    """
    Create a simple autoencoder model. Input is assumed to be an image
    :param input_shape: expects the input in format (height, width, n_channels)
    :return: dictionary with tensors required to train and evaluate the model
    """
    h, w, c = input_shape

    # %%%

    input_im = tf.placeholder(tf.float32, shape=[None, h, w, c])

    encoder_h1 = tf.Variable(tf.random_normal([h, w, c, minibatch_size])),
    encoder_b1 = tf.Variable(tf.random_normal([minibatch_size])),                                #bias

    decoder_h1 = tf.Variable(tf.random_normal([latent_space_size, minibatch_size])),
    decoder_b1 = tf.Variable(tf.random_normal([minibatch_size])),                                #bias

    encoding = tf.nn.sigmoid_cross_entropy_with_logits(tf.add(tf.matmul(input_im, encoder_h1), encoder_b1))
    decode = tf.nn.sigmoid_cross_entropy_with_logits(tf.add(tf.matmul(input_im, decoder_h1), decoder_b1))

    # Prediction
    y_pred = decode

    # Label
    y_true = input_im

    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    cost = loss

    # %%%

    model = {'cost': cost,
             'input': input,
             'enc': encoding,
             'dec': decode
             }
    return model


# Create model and tensors for evaluation
input_shape = (28, 28, 1)
model = create_model(input_shape)

evaluation = create_evaluation_tensor(model, evaluation_shape)

# Create optimizer
opt = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

# Create tensors for visualizing with tensorboard
# https://www.tensorflow.org/programmers_guide/saved_model
saver = tf.train.Saver()
for_tensorboard = create_summary_and_projector(model, evaluation, evaluation_set, run_path)

tf.set_random_seed(1)
with tf.Session() as sess:
    # Save graph
    # https: // www.tensorflow.org / programmers_guide / graph_viz
    train_writer = tf.summary.FileWriter(run_path, sess.graph)

    print("Initializing model")
    sess.run(tf.global_variables_initializer())

    for e in range(num_epochs):
        # iterate through minibatches
        for mb in mbs:
            batch_cost, _ = sess.run([model['cost'], opt],
                                     feed_dict={model['input']: mb[0]})

        # write current results to log
        write_to_tensorboard(sess, train_writer, for_tensorboard, evaluation_set, evaluation, e)
        # save trained model
        saver.save(sess, os.path.join(run_path, "model.ckpt"))

        print("Epoch: {}/{}".format(e+1, num_epochs),
              "batch cost: {:.4f}".format(batch_cost))
