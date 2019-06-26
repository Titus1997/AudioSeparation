# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:56:59 2019

@author: Titus
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:53:13 2019

@author: Titus
"""
import tensorflow as tf
from model import Model
from datas import Datas
from preprocess import to_spectrogram, get_magnitude, get_phase, to_wav_mag_only, soft_time_freq_mask, to_wav, write_wav
from config import EvalConfig, RunConfig

def get_drum(filename):
    
    tf.reset_default_graph()
    model = Model()
    
    
    with tf.Session(config=EvalConfig.session_conf) as sess:

        # Initialized, Load state
        sess.run(tf.global_variables_initializer())
        
        data = Datas(RunConfig.DATA_ROOT)
        model.load_state(sess, EvalConfig.CKPT_PATH)
        
        mixed_wav = data.get_mixture(filename)

        print(mixed_wav)
        
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
        pred_src2_mag = mixed_mag * mask_src2

        pred_src2_wav = to_wav(pred_src2_mag, mixed_phase)
        
        filename = filename.replace('.wav', '')
        write_wav(pred_src2_wav[0], '{}/{}'.format(RunConfig.RESULT_PATH, filename))

if __name__ == '__main__':
    get_drum('mixture.wav')
