import tensorflow as tf


def hello_world():
    w = tf.Variable(tf.ones(shape=(2, 2)), name="w")
    b = tf.Variable(tf.zeros(shape=2), name="b")

    @tf.function
    def forward(x):
        return w * x + b

    out_a = forward([1, 0])
    print(out_a)
