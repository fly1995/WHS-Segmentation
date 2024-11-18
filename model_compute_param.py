import numpy as np
from tensorflow.python import pywrap_tensorflow
import os
import tensorflow as tf


ckpt = tf.train.get_checkpoint_state(r"E:\D6_MMWHS_2017\MedicalDataAugmentationTool-MMWHS\output\Proposed_ct_1\2024-11-15_15-08-16\weights\\").model_checkpoint_path
saver = tf.train.import_meta_graph(ckpt+".meta")
variables = tf.trainable_variables()
total_parameters = 0
for variable in variables:
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)