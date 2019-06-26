# -*- coding: utf-8 -*-

from model import Model
import tensorflow as tf
import os
import shutil
from datas import Datas
from preprocess import to_spectrogram, get_magnitude
from config import TrainConfig

import matplotlib as plt
import librosa.display

# TODO multi-gpu
def train():
    tf.reset_default_graph()
    # Model
    model = Model()

    # Loss, Optimizer
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    loss_fn = model.loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=TrainConfig.LR).minimize(loss_fn, global_step=global_step)

    # Summaries
    summary_op = summaries(model, loss_fn)

    with tf.Session(config=TrainConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, TrainConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(TrainConfig.GRAPH_PATH, sess.graph)

        # Input source
        data = Datas(TrainConfig.DATA_PATH)

        for step in range(global_step.eval(), TrainConfig.FINAL_STEP): # changed xrange to range for py3
            mixed_wav, src1_wav, src2_wav = data.next_wav(TrainConfig.SECONDS)

            mixed_spec = to_spectrogram(mixed_wav)
            mixed_mag = get_magnitude(mixed_spec)

            src1_spec, src2_spec = to_spectrogram(src1_wav), to_spectrogram(src2_wav)
            src1_mag, src2_mag = get_magnitude(src1_spec), get_magnitude(src2_spec)

            src1_batch, _ = model.spec_to_batch(src1_mag)
            src2_batch, _ = model.spec_to_batch(src2_mag)
            mixed_batch, _ = model.spec_to_batch(mixed_mag)

            l, _, summary = sess.run([loss_fn, optimizer, summary_op],
                                     feed_dict={model.x_mixed: mixed_batch, model.y_src1: src1_batch,
                                                model.y_src2: src2_batch})

            print('For iteration {}\\t loss={}'.format(step, l))

            writer.add_summary(summary, global_step=step)

            # Save state
            if step % TrainConfig.CKPT_STEP == 0:
                tf.train.Saver().save(sess, TrainConfig.CKPT_PATH + '/checkpoint', global_step=step)

        writer.close()


def summaries(model, loss):
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('x_mixed', model.x_mixed)
    tf.summary.histogram('y_src1', model.y_src1)
    tf.summary.histogram('y_src2', model.y_src1)
    return tf.summary.merge_all()


def setup_path():
    if TrainConfig.RE_TRAIN:
        if os.path.exists(TrainConfig.CKPT_PATH):
            shutil.rmtree(TrainConfig.CKPT_PATH)
        if os.path.exists(TrainConfig.GRAPH_PATH):
            shutil.rmtree(TrainConfig.GRAPH_PATH)
    if not os.path.exists(TrainConfig.CKPT_PATH):
        os.makedirs(TrainConfig.CKPT_PATH)


if __name__ == '__main__':
    setup_path()
    train()