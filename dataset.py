# -*- coding: utf-8 -*-

import random
from os import walk
import os
from config import ModelConfig
from preprocess import get_random_wav, get_wav


class Dataset:
    def __init__(self, path):
        self.path = path

    def next_wavs(self, sec, size=1):
        wavfiles = []
        for (root, dirs, files) in walk(self.path):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wavfiles = random.sample(wavfiles, size)
        print(wavfiles)
        mixed, src1, src2 = get_random_wav(wavfiles, sec, ModelConfig.SR)
        return mixed, src1, src2, wavfiles
    
    def next_wav(self, sec):
        dir = ModelConfig.DATA_ROOT
        dirs = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
        wavdir = random.sample(dirs, 1)[0]
        mixed, src1, src2 = get_wav(wavdir, sec, ModelConfig.SR)
        return mixed, src1, src2
            