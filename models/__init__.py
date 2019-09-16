#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os

from .resnet50_fine import ResNet50_Fine

sys.path.append(os.pardir)

archs = {
         'resnet50_fine': ResNet50_Fine,
        }