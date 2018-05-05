"""Actor neural net"""

import tensorflow as tf


class Actor:
    """"Actor neural net. Approximate mu(s|theta) -> a"""
    def __init__(self, session, state_size, action_size):
        self.session = session # Tensorflow session
        self.state_size = state_size # number of elements forming a state
        self.action_size = action_size # number of elements forming an action

        # useful tensors
        self.input_state = None
        self.input_y = None
        self.output = None
        self.train = None
        self.create_neural_net()

    def create_neural_net(self):
        """Generate the neural net for actor and returns the tensors"""
        with tf.name_scope('actor'):
            self.input_state = tf.placeholder((None, self.state_size), tf.float32)
            self.input_y = tf.placeholder((None, self.action_size), tf.float32)

            l1 = tf.layers.dense(input=self.input_state, units=400,
                                 activation=tf.nn.relu, name='layer_1')
            l2 = tf.layers.dense(input=l1, units=300,
                                 activation=tf.nn.relu, name='layer_2')
            logits = tf.layers.dense(input=l2, units=self.action_size,
                                     activation=None, name='logits')
            self.output = tf.tanh(logits, name='output')

            # TODO check if resizing tanh to action space needed
            loss = tf.reduce_mean(tf.square(tf.subtract(self.output - self.input_y))) / 2

            self.train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    def pick_action(self, batch_state):
        """Choose an action"""
        result = self.session.run(self.output, feed_dict={self.input_state: batch_state})
        return result

    def train(self, batch_state, batch_y):
        self.session.run(self.train, feed_dict={self.input_state: batch_state,
                                                self.input_y: batch_y})
