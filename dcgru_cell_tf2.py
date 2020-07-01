import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations

from lib.matrix_calc import *


class DCGRUCell(keras.layers.Layer):

    def __init__(self, units, adj_mx, K_diffusion, num_nodes, filter_type, **kwargs):
        self.units = units
        self.state_size = units * num_nodes
        self.K_diffusion = K_diffusion
        self.num_nodes = num_nodes
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')
        super(DCGRUCell, self).__init__(**kwargs)
        self.supports = []
        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self.supports.append(build_sparse_matrix(support))
            sup0 = support
            for k in range(2, self.K_diffusion + 1):
                sup0 = support.dot(sup0)
                # sup0 = 2 * support.dot(sup0) - sup0     # (author version)
                self.supports.append(build_sparse_matrix(sup0))

    def build(self, input_shape):

        ## To get the kernel dimension we need to know how many graph convolution
        ## operations will be executed per gate

        self.num_mx = 1 + len(self.supports)  # * self.K_diffusion
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # ones before
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # ones before
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate):
        # print('inputs shape', inputs.get_shape())
        # batch_size = inputs.get_shape()[0] #.value
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        # inputs = tf.reshape(inputs, (batch_size, self.num_nodes, self.input_dim))
        state = tf.reshape(state, (-1, self.num_nodes, self.units))
        # state = tf.reshape(state, (batch_size, self.num_nodes, self.units))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]
        # print('input_size',input_size)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        # x0 = tf.reshape(x0, shape=[self.num_nodes, input_size * batch_size])
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)

        for support in self.supports:
            x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = tf.expand_dims(x_support, 0)
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
        x = tf.reshape(x, shape=[-1, input_size * self.num_mx])

        if gate == 'reset':
            x = tf.matmul(x, self.r_kernel)
            x = tf.nn.bias_add(x, self.r_bias)
        elif gate == 'update':
            x = tf.matmul(x, self.u_kernel)
            x = tf.nn.bias_add(x, self.u_bias)
        elif gate == 'candidate':
            x = tf.matmul(x, self.c_kernel)
            x = tf.nn.bias_add(x, self.c_bias)
        else:
            print('Error: Unknown gate')

        # print('x return shape', [batch_size, self.num_nodes * self.units])

        return tf.reshape(x, [-1, self.num_nodes * self.units])
