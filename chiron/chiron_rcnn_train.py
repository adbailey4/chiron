#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:32:32 2017

@author: haotianteng
"""
from __future__ import print_function
import tensorflow as tf
from distutils.dir_util import copy_tree
from chiron_input import read_raw_data_sets
from chiron_eval import sparse2dense, index2base
from utils.easy_assembler import simple_assembly
import numpy as np
from cnn import getcnnfeature
# from cnn import getcnnlogit
from rnn import rnn_layers
from nanotensor.run_nanotensor import test_for_nvidia_gpu, average_gradients

from nanotensor.utils import merge_two_dicts

import sys

# from rnn import rnn_layers_one_direction


def save_model():
    copy_tree(FLAGS.home_dir + '/chiron/chiron', FLAGS.log_dir + FLAGS.model_name + '/model')


def inference(x, seq_length, training, reuse=False):
    cnn_feature = getcnnfeature(x, training=training)
    feashape = cnn_feature.get_shape().as_list()
    # print(type(FLAGS.sequence_len))
    ratio = FLAGS.sequence_len / feashape[1]
    # print("ratio", ratio)
    logits = rnn_layers(cnn_feature, seq_length / ratio, training, class_n=FLAGS.bases ** FLAGS.k_mer + 1, reuse=reuse)
    #    logits = rnn_layers_one_direction(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 )
    #    logits = getcnnlogit(cnn_feature)
    return logits, ratio


def loss(logits, seq_len, label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(inputs=logits, labels=label, sequence_length=seq_len, ctc_merge_repeated=True, time_major=False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss', loss)
    return loss


def train_step(loss):
    opt = tf.train.AdamOptimizer(FLAGS.step_rate).minimize(loss)
    #    opt = tf.train.GradientDescentOptimizer(FLAGS.step_rate).minimize(loss)
    #    opt = tf.train.RMSPropOptimizer(FLAGS.step_rate).minimize(loss)
    #    opt = tf.train.MomentumOptimizer(FLAGS.step_rate,0.9).minimize(loss)
    return opt


def prediction(logits, seq_length, label, top_paths=1):
    """
    Args:
        logits:Input logits from a RNN.Shape = [batch_size,max_time,class_num]
        seq_length:sequence length of logits. Shape = [batch_size]
        label:Sparse tensor of label.
        top_paths:The number of top score path to choice from the decorder.
    """
    logits = tf.transpose(logits, perm=[1, 0, 2])
    predict = tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False, top_paths=top_paths)[0]
    predict_greedy = tf.nn.ctc_greedy_decoder(logits, seq_length, merge_repeated=True)

    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d, axis=0)
    d_min = tf.reduce_min(edit_d, axis=0)
    error = tf.reduce_mean(d_min, axis=0)
    tf.summary.scalar('Error_rate', error)
    return error, predict_greedy, predict, edit_d


def train(valid_reads_num=100):
    gpu_indexes = test_for_nvidia_gpu(FLAGS.num_gpu)
    tower_grads = []
    opt = tf.train.AdamOptimizer(FLAGS.step_rate)
    reuse = False
    batch_sizes = [10,20,30,100]
    sequence_lengths = [400, 300, 200, 100]
    if gpu_indexes:
        print("Using GPU's {}".format(gpu_indexes), file=sys.stderr)
        #with tf.variable_scope(tf.get_variable_scope()):
	for index, gpu in enumerate(gpu_indexes):
            with tf.variable_scope("my_model", reuse=reuse):
                training = tf.placeholder(tf.bool)
                x = tf.placeholder(tf.float32, shape=[batch_sizes[index], sequence_lengths[index]])
                seq_length = tf.placeholder(tf.int32, shape=[batch_sizes[index]])
                y_indexs = tf.placeholder(tf.int64)
                y_values = tf.placeholder(tf.int32) 
                y_shape = tf.placeholder(tf.int64)
                y = tf.SparseTensor(y_indexs, y_values, y_shape)

                with tf.device('/gpu:%d' % gpu):
                    logits, ratio = inference(x, seq_length, training)#, reuse=reuse)
                    # print(logits, ratio)
                    ctc_loss = loss(logits, seq_length, y)
                    #tf.get_variable_scope().reuse_variables()
                    reuse = True
                    gradients = opt.compute_gradients(ctc_loss)
                    tower_grads.append(gradients)
                    # print(gradients)
                    print(len(gradients))
        grads = average_gradients(tower_grads)
    else:
        print("No GPU's available, using CPU for computation", file=sys.stderr)
        training = tf.placeholder(tf.bool)
        x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
        seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        y_indexs = tf.placeholder(tf.int64)
        y_values = tf.placeholder(tf.int32)
        y_shape = tf.placeholder(tf.int64)
        y = tf.SparseTensor(y_indexs, y_values, y_shape)
        logits, ratio = inference(x, seq_length, training, reuse=False)
        # print(logits, ratio)
        ctc_loss = loss(logits, seq_length, y)
        grads = opt.compute_gradients(ctc_loss)
        print("Begin Loading Data. \n", file=sys.stderr)

        train_ds, valid_ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.sequence_len, valid_reads_num=valid_reads_num, k_mer=FLAGS.k_mer, alphabet=FLAGS.bases)



    train_op = opt.apply_gradients(grads)

    # print(logits, ratio)
    error, predict_greedy, predict, tmp_d = prediction(logits, seq_length, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # save_model()
    if FLAGS.retrain == False:
        sess.run(init)
        print("Model init finished. \n")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir + FLAGS.model_name))
        print("Model loaded finished. \n")
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir + FLAGS.model_name + '/summary/', sess.graph)



    for i in range(FLAGS.max_steps):
        if gpu_indexes:
            feed_dict = create_feed_dict(feed_tensors, train_datasets, batch_sizes)
        else:
            batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
            indxs, values, shape = batch_y
            feed_dict = {x: batch_x, seq_length: seq_len / ratio, y_indexs: indxs, y_values: values, y_shape: shape,
                        training: True}
        loss_val, _ = sess.run([ctc_loss, train_op], feed_dict=feed_dict)

        # answers = np.nonzero(np.negative(np.argmax(logits_val, axis=2)-4))
        # bpreads = [index2base(read) for read in predict_read]
        # print("bpreads", bpreads)

        if i % 100 == 0:
            valid_x, valid_len, valid_y = valid_ds.next_batch(FLAGS.batch_size)
            indxs, values, shape = valid_y
            feed_dict = {x: valid_x, seq_length: valid_len / ratio, y_indexs: indxs, y_values: values, y_shape: shape,
                         training: True}
            error_val = sess.run([error], feed_dict=feed_dict)
            print("Epoch %d, batch number %d, loss: %5.3f edit_distance: %5.3f"
                  % (train_ds.epochs_completed, train_ds.index_in_epoch, loss_val, error_val[0]))
            saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/model.ckpt', i)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

    saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/final.ckpt')

def create_feed_dict(tensor_list, datasets_list, batch_sizes, gpu_indexes):
    """Create feed dictionary from list of tensors and data sets"""
    feed_dict = {}
    for index, batch_size in enumerate(batch_sizes):
        dataset = datasets_list[index]
        dataset.next_batch(batch_size)

        batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
        indxs, values, shape = batch_y
        tensors = tensor_list[index]
        dict1 = {tensors[0]: batch_x, tensors[1]: seq_len / ratio, tensors[2]: indxs, tensors[3]: values, tensors[4]: shape,
                     tensors[5]: True}
        feed_dict = merge_two_dicts(dict1, feed_dict)

    return feed_dict

def run(args):
    global FLAGS
    FLAGS = args
    train(valid_reads_num=300)


if __name__ == "__main__":
    class Flags():
        def __init__(self):
            self.home_dir = "/Users/andrewbailey/CLionProjects/nanopore-RNN/test_files/minion-reads/test_methylated/"
            self.data_dir = self.home_dir
            self.log_dir = '/home/ubuntu/logs'
            self.model_name = 'logscrnn3+3-sep27'

            # self.log_dir = '/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/chiron/model/'
            # self.model_name = 'DNA_default'

            self.sequence_len = 200
            self.batch_size = 50
            self.step_rate = 1e-3
            self.max_steps = 1
            self.k_mer = 1
            self.bases = 5
            self.retrain = False
	    self.num_gpu = 4

    run(Flags())
