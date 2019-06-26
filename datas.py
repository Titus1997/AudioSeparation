# -*- coding: utf-8 -*-

import random
import os
from config import ModelConfig, RunConfig
from preprocess import get_wav, get_mixture


class Datas:
    def __init__(self, path):
        self.path = path
    
    def next_wav(self, sec):
        dir = ModelConfig.DATA_ROOT
        dirs = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
        wavdir = random.sample(dirs, 1)[0]
        mixed, src1, src2 = get_wav(os.path.abspath(self.path), wavdir, sec, ModelConfig.SR)
        return mixed, src1, src2
    
    def get_mixture(self, filename):
        mixture = get_mixture(os.path.abspath(self.path), filename)
        return mixture
