# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:05:22 2019

@author: Titus
"""

from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, GRUCell
from config import ModelConfig
import os

class Model:
    def __init__(self, no_rnn = 3, hidden_layer_size = 256):
        
        self.x_mixed = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME //2 + 1), name = 'x_mixed')
        self.y_src1 = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME //2 + 1), name = 'y_src1')
        self.y_src2 = tf.placeholder(tf.float32, shape=(None, None, ModelConfig.L_FRAME //2 + 1), name = 'y_src2')
        
        self.hidden_layer_size = hidden_layer_size
        self.n_rnn_layer = no_rnn
        self.net = tf.make_template('net', self._net)
        
        self()
        
    def __call__(self):
        return self.net()
    
    def _net(self):
        #lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_layer_size, state_is_tuple=True) # can use tf.nn.rnn_cell.GRUCell or tf.nn.rnn_cell.BasicRNNCell instead 
        rnn_layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_layer_size, state_is_tuple=True, name='lstmcell'+str(i)) for i in range(self.n_rnn_layer)], state_is_tuple=True)
        #rnn_layer = MultiRNNCell(GRUCell(self.hidden_layer_size) for _ in range(1, self.n_rnn_layer))
        rnn_output, rnn_state = tf.nn.dynamic_rnn(rnn_layer, self.x_mixed, dtype=tf.float32)
        
        s = self.x_mixed.get_shape()
        tup = tuple([s[i].value for i in range(0, len(s))])
        input_size = tup[2]
        
        y_dense_src1 = tf.layers.dense(inputs=rnn_output, units=input_size, activation=tf.nn.relu, name='y_dense_src1')
        y_dense_src2 = tf.layers.dense(inputs=rnn_output, units=input_size, activation=tf.nn.relu, name='y_dense_src2')

        #masking layers
        y_masking_src1 = y_dense_src1 / (y_dense_src1 + y_dense_src2 + np.finfo(float).eps) * self.x_mixed
        y_masking_src2 = y_dense_src2 / (y_dense_src1 + y_dense_src2 + np.finfo(float).eps) * self.x_mixed
        
        return y_masking_src1, y_masking_src2
    
    def loss(self):
        predicted_src1, predicted_src2 = self()
        return tf.reduce_mean(tf.square(self.y_src1 - predicted_src1) + tf.square(self.y_src2 - predicted_src2) , name = 'loss')

    
    @staticmethod
    # shape = (batch_size, n_freq, n_frames) => (batch_size, n_frames, n_freq)
    def spec_to_batch(src):
        num_wavs, freq, n_frames = src.shape

        # Padding
        pad_len = 0
        if n_frames % ModelConfig.SEQ_LEN > 0:
            pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
        pad_width = ((0, 0), (0, 0), (0, pad_len))
        padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

        assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)

        batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
        return batch, padded_src

    @staticmethod
    def batch_to_spec(src, num_wav):
        # shape = (batch_size, n_frames, n_freq) => (batch_size, n_freq, n_frames)
        batch_size, seq_len, freq = src.shape
        src = np.reshape(src, (num_wav, -1, freq))
        src = src.transpose(0, 2, 1)
        return src

    @staticmethod
    def load_state(sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)    
