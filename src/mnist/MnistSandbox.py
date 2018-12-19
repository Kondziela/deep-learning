import tensorflow as tf
import numpy as np


def random_batch(x_data, y_data, size):
    index_list = np.random.choice(np.arange(x_data.shape[0]), size)
    return x_data[index_list] / 255, y_data[index_list]


def convert_label(label):
    buff = np.zeros(10)
    buff[label] = 1
    return buff


def create_metric_variables():
    with tf.name_scope('losses'):
        metric_variables = dict()
        metric_variables['acc_train'] = tf.get_variable(shape=(), trainable=False, name='acc_train')
        metric_variables['acc_test'] = tf.get_variable(shape=(), trainable=False, name='acc_test')

        for name in metric_variables.keys():
            tf.summary.scalar(name, metric_variables[name])

        return metric_variables

###############

# NN PARAMS

###############

n_input = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10
learning_rate = 0.01
n_epochs = 40
batch_size = 250

################

# LOAD DATA

################

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape input
x_train = np.reshape(x_train, [x_train.shape[0], n_input])
x_test = np.reshape(x_test, [x_test.shape[0], n_input])


###############

# PLACEHOLDERS

###############

X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
Y = tf.placeholder(tf.int64, shape=(None), name="Y")

###############

# NN MODEL

##############

with tf.name_scope("NN"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="ukryta1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="ukryta2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_output, name="wyjscie")

#############

# LOSS FUNCTION

#############

with tf.name_scope("funkcja_straty"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)

    loss = tf.reduce_mean(xentropy, name="strata")

##############

# LEARNING FUNCTION

##############


with tf.name_scope("nauka"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_opt = optimizer.minimize(loss)

#############

# RATE FUNCTION

#############

with tf.name_scope("ocena"):
    correct = tf.nn.in_top_k(logits, Y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#############

# INIT AND START LEARNING

#############

init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    tb_writer = tf.summary.FileWriter('./tensorboard', sess.graph)
    summary_variables = create_metric_variables()
    merged_summaries = tf.summary.merge_all()
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(x_train.shape[0] // batch_size):
            X_batch, Y_batch = random_batch(x_train, y_train, batch_size)
            sess.run(training_opt, feed_dict={X: X_batch, Y: Y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})
        acc_test = accuracy.eval(feed_dict={X: x_test, Y: y_test})

        print(epoch, "Dokladnosc uczenia: ", acc_train, "Dokladnosc testowania: ", acc_test)

        sess.run(summary_variables['acc_train'].assign(acc_train))
        sess.run(summary_variables['acc_test'].assign(acc_test))

        tb_writer.add_summary(sess.run(merged_summaries), global_step=epoch)
        tb_writer.flush()


    save_path = saver.save(sess, './src/model/model_mnist.ckpt')
