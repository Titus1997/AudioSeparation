# -*- coding: utf-8 -*-


import os
import shutil

import numpy as np
import tensorflow as tf

from config import EvalConfig, ModelConfig
from dataset import Dataset
from model import Model
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only, soft_time_freq_mask, to_wav, write_wav


def eval():
    # Model
    model = Model()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.Session(config=EvalConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        model.load_state(sess, EvalConfig.CKPT_PATH)

        writer = tf.summary.FileWriter(EvalConfig.GRAPH_PATH, sess.graph)

        data = Dataset(EvalConfig.DATA_PATH)
        mixed_wav, src1_wav, src2_wav, wavfiles = data.next_wavs(EvalConfig.SECONDS, EvalConfig.NUM_EVAL)

        mixed_spec = to_spectrogram(mixed_wav)
        mixed_mag = get_magnitude(mixed_spec)
        mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
        mixed_phase = get_phase(mixed_spec)

        assert (np.all(np.equal(model.batch_to_spec(mixed_batch, EvalConfig.NUM_EVAL), padded_mixed_mag)))

        (pred_src1_mag, pred_src2_mag) = sess.run(model(), feed_dict={model.x_mixed: mixed_batch})

        seq_len = mixed_phase.shape[-1]
        pred_src1_mag = model.batch_to_spec(pred_src1_mag, EvalConfig.NUM_EVAL)[:, :, :seq_len]
        pred_src2_mag = model.batch_to_spec(pred_src2_mag, EvalConfig.NUM_EVAL)[:, :, :seq_len]

        # Time-frequency masking
        mask_src1 = soft_time_freq_mask(pred_src1_mag, pred_src2_mag)
        # mask_src1 = hard_time_freq_mask(pred_src1_mag, pred_src2_mag)
        mask_src2 = 1. - mask_src1
        pred_src1_mag = mixed_mag * mask_src1
        pred_src2_mag = mixed_mag * mask_src2

        # (magnitude, phase) -> spectrogram -> wav
        if EvalConfig.GRIFFIN_LIM:
            pred_src1_wav = to_wav_mag_only(pred_src1_mag, init_phase=mixed_phase, num_iters=EvalConfig.GRIFFIN_LIM_ITER)
            pred_src2_wav = to_wav_mag_only(pred_src2_mag, init_phase=mixed_phase, num_iters=EvalConfig.GRIFFIN_LIM_ITER)
        else:
            pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
            pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)

        # Write the result
        tf.summary.audio('GT_mixed', mixed_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
        tf.summary.audio('Pred_music', pred_src1_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)
        tf.summary.audio('Pred_vocal', pred_src2_wav, ModelConfig.SR, max_outputs=EvalConfig.NUM_EVAL)

        if EvalConfig.WRITE_RESULT:
            # Write the result
            for i in range(len(wavfiles)):
                name = wavfiles[i].replace('/', '-').replace('.wav', '')
                write_wav(mixed_wav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'original'))
                write_wav(pred_src1_wav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'music'))
                write_wav(pred_src2_wav[i], '{}/{}-{}'.format(EvalConfig.RESULT_PATH, name, 'voice'))

        writer.add_summary(sess.run(tf.summary.merge_all()), global_step=global_step.eval())

        writer.close()


def setup_path():
    if EvalConfig.RE_EVAL:
        if os.path.exists(EvalConfig.GRAPH_PATH):
            shutil.rmtree(EvalConfig.GRAPH_PATH)
        if os.path.exists(EvalConfig.RESULT_PATH):
            shutil.rmtree(EvalConfig.RESULT_PATH)

    if not os.path.exists(EvalConfig.RESULT_PATH):
        os.makedirs(EvalConfig.RESULT_PATH)


if __name__ == '__main__':
    setup_path()
    eval()