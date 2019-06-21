# -*- coding: utf-8 -*-

import tensorflow as tf

class ModelConfig:
    SR = 16000                # Sample Rate
    L_FRAME = 1024            # default 1024
    L_HOP = 256
    SEQ_LEN = 4
    # For Melspectogram
    N_MELS = 512
    F_MIN = 0.0
    DATA_ROOT = 'dataset/DSD100/Sources/Dev'

# Train
class TrainConfig:
    CASE = str(ModelConfig.SEQ_LEN) + 'frames_ikala'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    DATA_PATH = 'dataset/DSD100/Sources/Dev'
    LR = 0.01
    FINAL_STEP = 1000
    CKPT_STEP = 10
    NUM_WAVFILE = 1
    SECONDS = 8.192 # To get 512,512 in melspecto
    RE_TRAIN = True
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
    )


# TODO seperating model and case
# TODO config for each case
# Eval
class EvalConfig:
    # CASE = '1frame'
    # CASE = '4-frames-masking-layer'
    CASE = str(ModelConfig.SEQ_LEN) + 'frames_ikala'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    DATA_PATH = 'dataset/DSD100/ikala/Sources/Dev/051 - AM Contra - Heart Peripheral'
    # DATA_PATH = 'dataset/mir-1k/Wavfile'
    # DATA_PATH = 'dataset/ikala'
    GRIFFIN_LIM = False
    GRIFFIN_LIM_ITER = 1000
    NUM_EVAL = 9
    SECONDS = 60
    RE_EVAL = True
    EVAL_METRIC = False
    WRITE_RESULT = True
    RESULT_PATH = 'results/' + CASE
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False
    )