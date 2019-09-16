#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L
from chainer.functions.array.reshape import reshape


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link('a', BottleNeckA(in_size, ch, out_size, stride))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)

        return h

blockexpansion = 4



#### Attention Branch Network ####

class ResNet50(chainer.Chain):

    insize = 224

    def __init__(self, output=8):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(64)
            self.layer1 = Block(3, 64, 64, 256, 1)
            self.layer2 = Block(4, 256, 128, 512)
            self.layer3 = Block(6, 512, 256, 1024)

            self.att_layer4 = Block(3, 1024, 512, 2048, 1)
            self.bn_att = L.BatchNormalization(512*blockexpansion)
            self.att_conv = L.Convolution2D(
                512*blockexpansion, output, ksize=1, pad=0, initialW=initializers.HeNormal(), nobias=True)
            self.bn_att2 = L.BatchNormalization(output)
            self.att_conv2 = L.Convolution2D(
                output, output, ksize=1, pad=0, initialW=initializers.HeNormal(), nobias=True)
            self.att_conv3 = L.Convolution2D(
                output, 1, ksize=3, pad=1, initialW=initializers.HeNormal(), nobias=True)
            self.bn_att3 = L.BatchNormalization(1)

            self.layer4 = Block(3, 1024, 512, 2048)
            self.fc = L.Linear(512*blockexpansion, output)


    def __call__(self, x):

    #######################################################################
    #                        Feature extractor                            #
    #######################################################################

        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2, pad=1)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)

        fe = h

    #######################################################################
    #                        Attention branch                             #
    #######################################################################

        ah = self.bn_att(self.att_layer4(h))
        ah = F.relu(self.bn_att2(self.att_conv(ah)))
        self.att = F.sigmoid(self.bn_att3(self.att_conv3(ah)))
        ah = self.att_conv2(ah)
        #ah = F.average_pooling_2d(ah, 14)
        ah = _global_average_pooling_2d1(ah)

    #######################################################################
    #                        Perception branch                            #
    #######################################################################

        rh = h * self.att #ここでAttention Mapをかけて
        rh = rh + h # ここで足す
        per = rh
        rh = self.layer4(rh)
        #rh = F.average_pooling_2d(rh, 7, stride=1)
        rh = _global_average_pooling_2d2(rh)
        rh = self.fc(rh)

        return ah, rh, [self.att, fe, per]



def _global_average_pooling_2d1(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=2)
    h = reshape(h, (n, channel))
    return h

def _global_average_pooling_2d2(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h