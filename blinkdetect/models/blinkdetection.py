#!/usr/bin/env python3
# coding: utf-8

import torch
from torch.nn import (
    Sequential,
    Flatten, Conv1d, MaxPool1d, Linear,
    ReLU, Sigmoid, Dropout)
from collections import OrderedDict


class BlinkDetector(torch.nn.Module):
    
    def __init__(self, input_dim):
        super(BlinkDetector, self).__init__()

        self.conv_block = self.build_conv1d_block(input_dim, 64)

        self.regressor = Sequential(
            self.build_mlp(64, 1),
            Sigmoid()
        )

        self.classifier = self.build_mlp(64, 2)


    @staticmethod
    def build_mlp(input_dim, output_dim):

        return Sequential(
            OrderedDict([
                ('fc_conv', Linear(input_dim, 32)),
                ('relu_conv', ReLU()),
                ('dropout_out', Dropout(0.2)),
                ('output', Linear(32, output_dim)),]
            ))


    @staticmethod
    def build_conv1d_block(input_dim, output_dim):
        # ARCH test
        return Sequential(
            OrderedDict([
                ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
                ('relu_1', ReLU()),

                ('conv1d_2', Conv1d(32, 16, 3, 1)), # 26
                ('relu_2', ReLU()),

                ('maxpool_2', MaxPool1d(3,3)), # 8

                ('flatten', Flatten()),
                ('fc_conv', Linear(128, output_dim)),
            ]))


    def forward(self, x):
        conv_out = self.conv_block(x)
        classification = self.classifier(conv_out)
        duration = self.regressor(conv_out)
        return classification, duration
