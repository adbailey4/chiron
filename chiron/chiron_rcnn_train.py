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
    if gpu_indexes:
        print("Using GPU's {}".format(gpu_indexes), file=sys.stderr)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in list(gpu_indexes):
                with tf.variable_scope("my_model", reuse=(len(gpu_indexes))>1):
                    training = tf.placeholder(tf.bool)
                    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
                    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
                    y_indexs = tf.placeholder(tf.int64)
                    y_values = tf.placeholder(tf.int32)
                    y_shape = tf.placeholder(tf.int64)
                    y = tf.SparseTensor(y_indexs, y_values, y_shape)
                    with tf.device('/gpu:%d' % i):
                        logits, ratio = inference(x, seq_length, training, reuse=reuse)
                        # print(logits, ratio)
                        ctc_loss = loss(logits, seq_length, y)
                        tf.get_variable_scope().reuse_variables()
                        reuse = True
                        gradients = opt.compute_gradients(ctc_loss)
                        tower_grads.append(gradients)
                        # print(gradients)
                        # print(len(gradients))
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
        print("Model init finished, begin loading data. \n")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir + FLAGS.model_name))
        print("Model loaded finished, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir + FLAGS.model_name + '/summary/', sess.graph)

    train_ds, valid_ds = read_raw_data_sets(FLAGS.data_dir, FLAGS.sequence_len, valid_reads_num=valid_reads_num, k_mer=FLAGS.k_mer, alphabet=FLAGS.bases)
    # train_ds1, valid_ds1 = read_raw_data_sets(FLAGS.data_dir, 400, valid_reads_num=valid_reads_num, k_mer=FLAGS.k_mer)
    # train_ds2, valid_ds2 = read_raw_data_sets(FLAGS.data_dir, 600, valid_reads_num=valid_reads_num, k_mer=FLAGS.k_mer)
    # train_ds3, valid_ds3 = read_raw_data_sets(FLAGS.data_dir, 1000, valid_reads_num=valid_reads_num, k_mer=FLAGS.k_mer)

    for i in range(FLAGS.max_steps):
        batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
        indxs, values, shape = batch_y
        feed_dict = {x: batch_x, seq_length: seq_len / ratio, y_indexs: indxs, y_values: values, y_shape: shape,
                     training: True}
        loss_val, _ = sess.run([ctc_loss, train_op], feed_dict=feed_dict)
        # print("loss val", loss_val)
        # # print(predict_greedy)
        # print("batch_y", batch_y)
        # print("logits_val", logits_val)
        # print("logits_val_shape", logits_val.shape)
        # print(np.argmax(logits_val, axis=2))
        # answers = np.nonzero(np.negative(np.argmax(logits_val, axis=2)-4))
        # print("argmax", np.argmax(logits_val, axis=2))
        # print("argmax-4", np.argmax(logits_val, axis=2)-4)
        # print("npnegative-4",np.negative(np.argmax(logits_val, axis=2)-4))
        # print("answers", answers)
        # print("Final?", (np.argmax(logits_val, axis=2)-4)[answers]+4)
        # print("values", values)
        # print("indxs", indxs)
        # print("values", values)
        # print("shape", shape)
        # print("predict2", predict2)
        # print("tmp_d_val", tmp_d_val)
        # print(tf.edit_distance(predict2[0], tf.SparseTensor(indxs, values, shape)))
        # predict_read = sparse2dense(predict_greedy2)[0]
        # print("predict_read", predict_read)
        # predict_read = sparse2dense(predict2)[0]
        # # predict_read1 = sparse2dense(predict2)[0]
        # # print("read_len", len(predict_read))
        # print("predict_read", predict_read)

        # print("predict_read1", len(predict_read1))
        # print("predict_read", len(predict_read))

        # bpreads = [index2base(read) for read in predict_read]
        # print("bpreads", bpreads)
        # concensus = simple_assembly(bpreads)
        # c_bpread = index2base(np.argmax(concensus, axis=0))
        # print(c_bpread)
        if i % 10 == 0:
            valid_x, valid_len, valid_y = valid_ds.next_batch(FLAGS.batch_size)
            # print(valid_len)
            # print(valid_y)
            indxs, values, shape = valid_y
            # print(tf.shape(valid_x))
            # print(tf.shape(valid_y))
            # print(values)
            feed_dict = {x: valid_x, seq_length: valid_len / ratio, y_indexs: indxs, y_values: values, y_shape: shape,
                         training: True}
            error_val = sess.run([error], feed_dict=feed_dict)
            # print(loss_val)
            # print(error_val)
            print("Epoch %d, batch number %d, loss: %5.3f edit_distance: %5.3f"
                  % (train_ds.epochs_completed, train_ds.index_in_epoch, loss_val, error_val[0]))
            # print("predict_val", predict_val)
            # # print("predict_greedy_val", predict_greedy_val)
            # print("tf.to_int32(predict_val[i])", tf.to_int32(predict_val[0][0]))
            #
            # print("tmp_d_val", tmp_d_val)


            saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/model.ckpt', i)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

    saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/final.ckpt')


def run(args):
    global FLAGS
    FLAGS = args
    train(valid_reads_num=300)


if __name__ == "__main__":
    class Flags():
        def __init__(self):
            self.home_dir = "/home/ubuntu/data/methylated_ecoli/training/"
            self.data_dir = self.home_dir
            self.log_dir = '/home/ubuntu/logs'
            self.model_name = 'logscrnn3+3-sep27'

            # self.log_dir = '/Users/andrewbailey/CLionProjects/nanopore-RNN/chiron/chiron/model/'
            # self.model_name = 'DNA_default'

            self.sequence_len = 600
            self.batch_size = 50
            self.step_rate = 1e-3
            self.max_steps = 100000
            self.k_mer = 1
            self.bases = 5
            self.retrain = False
	    self.num_gpu = 1

    run(Flags())
