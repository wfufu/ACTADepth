#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ACTADepth 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：fyh
@Date    ：23/11/6 13:31 
"""
from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
