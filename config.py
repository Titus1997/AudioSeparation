# -*- coding: utf-8 -*-

import tensorflow as tf

class RunConfig:
    DATA_ROOT = 'audios'
    RESULT_PATH = 'outputs'

class ModelConfig:
    SR = 44100                # Sample Rate
    L_FRAME = 1024            # default 1024
    L_HOP = 256
    SEQ_LEN = 4
    # For Melspectogram
    N_MELS = 512
    F_MIN = 0.0
    #DATA_ROOT = 'dataset/DSD100/Sources/Dev'
    DATA_ROOT = 'dataset/eval'

# Train
class TrainConfig:
    CASE = 'dsd'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/train'
    #DATA_PATH = 'dataset/DSD100/Sources/Dev'
    DATA_PATH = 'dataset/eval'
    LR = 0.01
    FINAL_STEP = 10000
    CKPT_STEP = 600
    NUM_WAVFILE = 1
    SECONDS = 16.384 # To get 512,512 in melspecto
    RE_TRAIN = False
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.25
        ),
    )

# Eval
class EvalConfig:
    CASE = 'dsd'
    CKPT_PATH = 'checkpoints/' + CASE
    GRAPH_PATH = 'graphs/' + CASE + '/eval'
    #DATA_PATH = 'dataset/DSD100/Sources/Dev'
    DATA_PATH = 'dataset/eval'
    GRIFFIN_LIM = False
    GRIFFIN_LIM_ITER = 1000
    NUM_EVAL = 1
    SECONDS = 60
    RE_EVAL = False
    EVAL_METRIC = False
    WRITE_RESULT = True
    RESULT_PATH = 'results/' + CASE
    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 1},
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=False
    )