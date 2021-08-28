#!/usr/bin/env python3
# coding: utf-8

import torch
from torch.nn import Sequential, Flatten, Conv1d, MaxPool1d, ReLU, Linear, BatchNorm1d, Dropout, ELU, SELU, Sigmoid
from collections import OrderedDict




class BlinkDetector(torch.nn.Module):
    
    def __init__(self, input_dim):
        super(BlinkDetector, self).__init__()

        self.conv_block = self.build_conv1d_block(input_dim, 32)
        

        self.regressor = Sequential(
            self.build_mlp(32, 1),
            Sigmoid()
        )

        self.classifier = self.build_mlp(32, 1)

    @staticmethod
    def build_mlp(input_dim, output_dim):
        # Arch 1 -- Arch 2 -- Arch 3 -- Arch 3.4
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 64)),
        #                                ('relu_conv', ReLU()),
        #                             #    ('dropout', Dropout(0.5)),
        #                             #    ('fc_1', Linear(128, 64)),
        #                             #    ('relu1', ReLU()),
        #                             #    ('dropout1', Dropout(0.5)),
        #                                ('fc_2', Linear(64, 32)),
        #                                ('relu2', ReLU()),
        #                                ('dropout2', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # Arch 3.1.4
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 128)),
        #                               ('relu_conv', ELU()),
        #                                ('fc_1', Linear(128, 64)),
        #                                ('relu1', ELU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('fc_2', Linear(64, 32)),
        #                                ('relu2', ELU()),
        #                                ('dropout2', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # # Arch 3.1 -- Arch 3.1.1 -- 3.1.2 -- 3.1.3
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 64)),
        #                                ('relu_conv', ReLU()),
        #                                ('fc_1', Linear(64, 32)),
        #                                ('relu_1', ReLU()),
        #                                ('dropout', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim)),]
        #                                ))

        # Arch test
        return Sequential(OrderedDict([
                                    #    ('dropout_0', Dropout(0.2)),
                                    #    ('fc_conv', Linear(input_dim, 32)),
                                    #    ('relu_conv', ReLU()),
                                    #    ('dropout', Dropout(0.2)),
                                    #    ('fc_1', Linear(64, 32)),
                                    #    ('relu1', ReLU()),
                                       ('dropout_out', Dropout(0.2)),
                                       ('output', Linear(input_dim, output_dim)),]
                                    #    ('relu1', ELU()),
                                    #    ('dropout2', Dropout(0.2)),
                                    #    ('output', Linear(32, output_dim))]
                                      ))

        # Arch 3.2 -- Arch 2.2
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 128)),
        #                               ('relu_conv', ReLU()),
        #                                ('fc_1', Linear(128, 64)),
        #                                ('relu1', ReLU()),
        #                                ('fc_2', Linear(64, 32)),
        #                                ('relu2', ReLU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # Arch 2.2.1
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 128)),
        #                               ('relu_conv', ReLU()),
        #                               ('dropout1', Dropout(0.2)),
        #                                ('fc_1', Linear(128, 64)),
        #                                ('relu1', ReLU()),
        #                                ('dropout2', Dropout(0.2)),
        #                                ('fc_2', Linear(64, 32)),
        #                                ('relu2', ReLU()),
        #                                ('dropout3', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # Arch 4 - 4.1
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 64)),
        #                                ('relu_conv', ReLU()),
        #                                ('fc_1', Linear(64, 32)),
        #                                ('relu1', ReLU()),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # Arch 4.1.0
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 64)),
        #                                ('relu_conv', ReLU()),
        #                             #    ('dropout', Dropout(0.2)),
        #                                ('fc_1', Linear(64, 32)),
        #                                ('relu1', ReLU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))
        

        # Arch 4.2 -- 4.1.1 -- 4.1.2
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 64)),
        #                                ('relu_conv', ReLU()),
        #                                ('dropout', Dropout(0.2)),
        #                                ('fc_1', Linear(64, 32)),
        #                                ('relu1', ReLU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # Arch 5
        # return Sequential(OrderedDict([
        #                                ('dropout1', Dropout(0.2)),
        #                                ('fc_conv', Linear(input_dim, 32)),
        #                                ('relu_conv', ReLU()),
        #                                ('output', Linear(32, output_dim))]
        #                               ))
        
        # Arch 5.1
        # return Sequential(OrderedDict([
        #                                ('dropout1', Dropout(0.2)),
        #                                ('fc_conv', Linear(input_dim, 32)),
        #                                ('relu_conv', ReLU()),
        #                                ('output', Linear(32, output_dim))]
        #                               ))
                                      
        # Arch 5.2 - Arch 6
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 32)),
        #                                ('relu_conv', ReLU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

        # Arch 6.1
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 32)),
        #                                ('relu_conv', ReLU()),
        #                               #  ('dropout1', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))  
        
        # Arch 6.3
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 32)),
        #                                ('relu_conv', ReLU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))  

        # Arch 6.2
        # return Sequential(OrderedDict([
        #                                ('output', Linear(input_dim, output_dim)),
        #                               #  ('fc_conv', Linear(input_dim, 32)),
        #                               #  ('relu_conv', ReLU()),
        #                               #  ('dropout1', Dropout(0.2)),
        #                               #  ('output', Linear(32, output_dim))
        #                                ]
        #                               ))  

        # Arch 3.3
        # return Sequential(OrderedDict([
        #                                ('fc_conv', Linear(input_dim, 128)),
        #                               ('relu_conv', ReLU()),
        #                                ('fc_1', Linear(128, 64)),
        #                                ('relu1', ReLU()),
        #                                ('dropout1', Dropout(0.2)),
        #                                ('fc_2', Linear(64, 32)),
        #                                ('relu2', ReLU()),
        #                                ('dropout2', Dropout(0.2)),
        #                                ('output', Linear(32, output_dim))]
        #                               ))

    @staticmethod
    def build_conv1d_block(input_dim, output_dim):
        # ARCH 1
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), # 28
        #                   ('relu1', ReLU()),
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 26
        #                   ('relu2', ReLU()),
        #                   ('maxpool_2', MaxPool1d(2,2)), # 13
        #                 #   
        #                   ('conv1d_3', Conv1d(16, 8, 3, 1)), # 11
        #                   ('relu3', ReLU()),
                          
        #                   ('conv1d_4', Conv1d(8, 4, 3, 1)), # 9
        #                   ('relu4', ReLU()),

        #                   ('maxpool_4', MaxPool1d(2,2)), # 4

        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(16, output_dim)),
        #                   ]))

        # ARCH 2 -- Arch 2.2
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)),
        #                   ('batchnorm1_1', BatchNorm1d(32)), # original paper before the activation layer
        #                   ('dropout1', Dropout(0.2)),
        #                   ('relu1', ReLU()),
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)),
        #                   ('relu2', ReLU()),
        #                   ('conv1d_3', Conv1d(16, 8, 3, 1)),
        #                   ('relu3', ReLU()),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(192, output_dim)),
        #                   ]))

        # ARCH 2.1
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)),
        #                   ('batchnorm1_1', BatchNorm1d(32)), # original paper before the activation layer
        #                   ('relu1', ReLU()),
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)),
        #                   ('batchnorm1_2', BatchNorm1d(16)),
        #                   ('relu2', ReLU()),
        #                   ('conv1d_3', Conv1d(16, 8, 3, 1)),
        #                   ('batchnorm1_3', BatchNorm1d(8)),
        #                   ('relu3', ReLU()),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(192, output_dim)),
        #                   ]))
        
        # ARCH test
        return Sequential(OrderedDict([
                          ('conv1d_1', Conv1d(input_dim, 16, 3, 1)), #28
                          ('relu_1', ReLU()),
                          
                          ('conv1d_2', Conv1d(16, 8, 3, 1)), # 26
                          ('relu_2', ReLU()),
                          ('maxpool_2', MaxPool1d(3,3)), # 8

                        #   ('conv1d_3', Conv1d(16, 8, 3, 1)), # 11
                        #   ('relu_3', ReLU()),
                        #   ('maxpool_3', MaxPool1d(2,2)), # 5

                          ('flatten', Flatten()),
                        #   ('dropout', Dropout(0.2)),
                          ('fc_conv', Linear(64, output_dim)),
                          
                          ]))

        # ARCH 3 -- 3.1 -- 3.1.4
        return Sequential(OrderedDict([
                          ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
                          ('relu1', ReLU()),
                        #   ('dropout', Dropout(0.2)),
                          ('maxpool_1', MaxPool1d(2,2)), # 14 # before or after activation does not matter
                          ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
                          ('relu2', ReLU()),
                          ('flatten', Flatten()),
                          ('fc_conv', Linear(192, output_dim)),
                          ]))

        # ARCH 3.1.1
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm1_1', BatchNorm1d(32)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # (28 - 2)/2 + 1 = 14 # before or after activation does not matter
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('relu2', ReLU()),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(192, output_dim)),
        #                   ]))

        # ARCH 3.1.2
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm1_1', BatchNorm1d(32)),
        #                   ('dropout1', Dropout(0.2)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # (28 - 2)/2 + 1 = 14 # before or after activation does not matter
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('relu2', ReLU()),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(192, output_dim)),
        #                   ]))

        # ARCH 3.1.3
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm1_1', BatchNorm1d(32)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # (28 - 2)/2 + 1 = 14 # before or after activation does not matter
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('batchnorm1_1', BatchNorm1d(32)),
        #                   ('relu2', ReLU()),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(192, output_dim)),
        #                   ]))


        # ARCH 3.4
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm1_1', BatchNorm1d(32)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # (28 - 2)/2 + 1 = 14 # before or after activation does not matter
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('relu2', ReLU()),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(192, output_dim)),
        #                   ]))


        # ARCH 4 -- 4.1.0 -- 4.1.1
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # 14
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('relu2', ReLU()),
        #                   ('maxpool_2', MaxPool1d(2,2)),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(96, output_dim)),
        #                   ]))

        # Arch 4.1.2
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # 14
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('relu2', ReLU()),
        #                   ('avgpool_2', AvgPool1d(2,2)),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(96, output_dim)),
        #                   ]))
        
        

        # ARCH 4.1 -- 4.2
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm_1', BatchNorm1d(32)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # 14
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('batchnorm_2', BatchNorm1d(16)),
        #                   ('relu2', ReLU()),
        #                   ('maxpool_2', MaxPool1d(2,2)),
        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(96, output_dim)),
        #                   ]))

        # Arch 5 -- 5.1 -- 5.2
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm_1', BatchNorm1d(32)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # 14
                          
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   ('batchnorm_2', BatchNorm1d(16)),
        #                   ('relu2', ReLU()),
        #                   ('maxpool_2', MaxPool1d(2,2)),

        #                   ('conv1d_3', Conv1d(16, 8, 3, 1)), # 4
        #                   ('batchnorm_3', BatchNorm1d(8)),
        #                   ('relu3', ReLU()),

        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(32, output_dim)),
        #                   ]))
        
        # Arch 6 -- 6.1 -- 6.2 -- 6.3
        # return Sequential(OrderedDict([
        #                   ('conv1d_1', Conv1d(input_dim, 32, 3, 1)), #28
        #                   ('batchnorm_1', BatchNorm1d(32)),
        #                   ('relu1', ReLU()),
        #                   ('maxpool_1', MaxPool1d(2,2)), # 14
                          
        #                   ('conv1d_2', Conv1d(32, 16, 3, 1)), # 12
        #                   # ('batchnorm_2', BatchNorm1d(16)),
        #                   ('relu2', ReLU()),
        #                   ('maxpool_2', MaxPool1d(2,2)),

        #                   ('conv1d_3', Conv1d(16, 8, 3, 1)), # 4
        #                   # ('batchnorm_3', BatchNorm1d(8)),
        #                   ('relu3', ReLU()),

        #                   ('flatten', Flatten()),
        #                   ('fc_conv', Linear(32, output_dim)),
        #                   ]))

        

    def forward(self, x):
        conv_out = self.conv_block(x)
        classification = self.classifier(conv_out)
        duration = self.regressor(conv_out)
        return classification, duration