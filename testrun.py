# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:53:13 2019

@author: Titus
"""
import tensorflow as tf
import os
import shutil
from model import Model
from dataset import Dataset
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only, soft_time_freq_mask, to_wav, write_wav
from config import TrainConfig, EvalConfig

def test_run():
    
    model = Model()
    
    
    with tf.Session(config=EvalConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        
        data = Dataset(EvalConfig.DATA_PATH)
        
        model.load_state(sess, TrainConfig.CKPT_PATH)
        
        mixed_wav, src1_wav, src2_wav = data.next_wav(EvalConfig.SECONDS)

        
        mixed_spec = to_spectrogram(mixed_wav)
        mixed_mag = get_magnitude(mixed_spec)
        mixed_batch, padded_mixed_mag = model.spec_to_batch(mixed_mag)
        mixed_phase = get_phase(mixed_spec)

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

        pred_src1_wav = to_wav(pred_src1_mag, mixed_phase)
        pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)
        
        write_wav(mixed_wav[0], '{}/{}'.format(EvalConfig.RESULT_PATH, 'original'))
        write_wav(pred_src1_wav[0], '{}/{}'.format(EvalConfig.RESULT_PATH, 'music'))
        write_wav(pred_src2_wav[0], '{}/{}'.format(EvalConfig.RESULT_PATH, 'voice'))

if __name__ == '__main__':
    test_run()