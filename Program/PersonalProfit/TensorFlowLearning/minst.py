from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def comput_graph(mnist):
    x = tf.placeholder('float', [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(500)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        #print(sess.run(cross_entropy,feed_dict={x:batch_xs, y_:batch_ys}))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def mygraph(mnist):
    x = tf.placeholder('float',[None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros(10))
    y = tf.nn.softmax(tf.matmul(x,w)+b)
    y_true = tf.placeholder('float', [None, 10])
    cross_entropy = -tf.reduce_sum(y_true*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_xs, y_true: batch_ys})
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('data/', one_hot=True)
    mygraph(mnist)
