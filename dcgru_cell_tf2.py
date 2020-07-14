"""
Alternative implementation of the DCGRU recurrent cell in Tensorflow 2
References
----------
Paper: https://arxiv.org/abs/1707.01926
Original implementation: https://github.com/liyaguang/DCRNN
"""

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
        # the formula describing the diffsuion convolution operation in the paper
        # corresponds to the filter "dual_random_walk"
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
                sup0 = support.dot(sup0)                  # (original paper version)
                # sup0 = 2 * support.dot(sup0) - sup0     # (author's repository version)
                self.supports.append(build_sparse_matrix(sup0))

    def build(self, input_shape):
        """
        Defines kernel and biases of the DCGRU cell.
        To get the kernel dimension we need to know how many graph convolution
        operations will be executed per gate, hence the number of support matrices
        (+1 to account for the input signal itself).
        
        input_shape: (None, num_nodes, input_dim)
        """
        self.num_mx = 1 + len(self.supports)
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units) * self.num_mx

        self.r_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='r_kernel')
        self.r_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='r_bias')

        self.u_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='u_kernel')
        self.u_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',  # originally ones
                                      name='u_bias')

        self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
                                        initializer='glorot_uniform',
                                        name='c_kernel')
        self.c_bias = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='c_bias')

        self.built = True

    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, num_nodes, input_dim)
        states[0]: (batch_size, num_nodes * units)
        """
        h_prev = states[0]

        r = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'reset'))
        u = self.recurrent_activation(self.diff_conv(inputs, h_prev, 'update'))
        c = self.activation(self.diff_conv(inputs, r * h_prev, 'candidate'))

        h = u * h_prev + (1 - u) * c

        return h, [h]

    def diff_conv(self, inputs, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, num_nodes, input_dim)
        state: (batch_size, num_nodes * units)
        gate: "reset", "update", "candidate"
        """
        assert inputs.get_shape()[1] == self.num_nodes
        assert inputs.get_shape()[2] == self.input_dim
        state = tf.reshape(state, (-1, self.num_nodes, self.units)) # (batch_size, num_nodes, units)
        # concatenate inputs and state
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2]    # (input_dim + units)

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, input_dim + units, batch_size)
        x0 = tf.reshape(x0, shape=[self.num_nodes, -1])
        x = tf.expand_dims(x0, axis=0)
        
        for support in self.supports:
            # premultiply the concatened inputs and state with support matrices
            x_support = tf.sparse.sparse_dense_matmul(support, x0)
            x_support = tf.expand_dims(x_support, 0)
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=0)

        x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
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

        return tf.reshape(x, [-1, self.num_nodes * self.units]) # (batch_size, num_nodes * units)
