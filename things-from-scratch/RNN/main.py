import tensorflow as tf

class RNN(tf.keras.layers.Layer):
    def add_weights(self,shape):
        return tf.Variable(tf.random.normal(shape,stddev=0.01))
    def __init__(self,rnn_units,input_dim,output_dim):
        super(RNN,self).__init__()

        self.Wxh = self.add_weights([rnn_units,input_dim])
        self.Whh = self.add_weights([rnn_units,rnn_units])
        self.Why = self.add_weights([output_dim,rnn_units])

        self.h = tf.zeros([rnn_units,1])

    def call(self, x):
        self.h = tf.math.tanh(tf.matmul(self.Whh, self.h) + tf.matmul(self.Wxh, x))
        outputs = tf.matmul(self.Why, self.h)
        return outputs, self.h

rnn = RNN(64,10,1)
x = tf.random.normal([10,1])
output, h = rnn.call(x)
print(output.shape)
print(h.shape)

