import tensorflow.compat.v1 as tf
from tensorflow_train.layers.layers import conv3d, concat_channels, avg_pool3d, conv3d_transpose, add,max_pool3d, upsample3d,dense,dropout
from tensorflow_train.layers.interpolation import upsample3d_linear
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.layers.initializers import he_initializer
from tensorflow_train.layers.normalizers import batch_norm
from tensorflow.python.keras.layers import Activation, BatchNormalization, Conv3DTranspose, Concatenate, Add, Dropout,Dense,Embedding
from keras.optimizers import Adam
from keras.layers import Lambda, Reshape, Input, concatenate, Conv3D,Dropout,BatchNormalization, add, Concatenate, Activation,GlobalAveragePooling3D, Reshape,Add,multiply,GlobalMaxPooling3D
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import numpy as np
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention



class UnetClassicAvgLinear3d(UnetBase):
    def combine(self, parallel_node, upsample_node, current_level, is_training):
        return concat_channels([parallel_node, upsample_node], name='concat' + str(current_level), data_format=self.data_format)

    def contracting_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def parallel_block(self, node, current_level, is_training):
        return node

    def expanding_block(self, node, current_level, is_training):
        node = self.conv(node, current_level, '0', is_training)
        node = self.conv(node, current_level, '1', is_training)
        return node

    def downsample(self, node, current_level, is_training):
        return avg_pool3d(node, [2, 2, 2], name='downsample' + str(current_level), data_format=self.data_format)

    def upsample(self, node, current_level, is_training):
        return upsample3d_linear(node, [2, 2, 2], name='upsample' + str(current_level), data_format=self.data_format)

    def conv(self, node, current_level, postfix, is_training):
        return conv3d(node,
                      self.num_filters(current_level),
                      [3, 3, 3],
                      name='conv' + postfix,
                      activation=self.activation,
                      normalization=None,
                      is_training=is_training,
                      data_format=self.data_format,
                      kernel_initializer=self.kernel_initializer,
                      padding=self.padding)


def network_scn(input, num_labels, is_training, data_format='channels_first'):
    downsampling_factor = 4
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = he_initializer
    spatial_activation = None
    padding = 'reflect'
    with tf.variable_scope('unet'):#关于输入输出大小的备注，以CT为例，MRI以此类推
        unet = UnetClassicAvgLinear3d(64, 4, data_format=data_format, double_filters_per_level=True, kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        local_prediction = unet(input, is_training=is_training)
        print(input.shape)#(1, 1, 64, 64, 64)
        local_prediction = conv3d(local_prediction, num_labels, [1, 1, 1], name='local_prediction', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
        print(local_prediction.shape)#(1, 8, 64, 64, 64)
    with tf.variable_scope('spatial_configuration'):
        local_prediction_pool = avg_pool3d(local_prediction, [downsampling_factor] * 3, name='local_prediction_pool')#[downsampling_factor] * 3 相当于kernel_size [4, 4, 4]
        scconv = conv3d(local_prediction_pool, 64, [5, 5, 5], name='scconv0', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [5, 5, 5], name='scconv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [5, 5, 5], name='scconv2', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        spatial_prediction_pool = conv3d(scconv, num_labels, [5, 5, 5], name='spatial_prediction_pool', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)
        #spatial_prediction_pool: in=[1, 64, 16, 16, 16] out=[1, 8, 16, 16, 16]
        spatial_prediction = upsample3d_linear(spatial_prediction_pool, [downsampling_factor] * 3, name='spatial_prediction', padding='valid_cropped')
    with tf.variable_scope('combination'):
        prediction = local_prediction * spatial_prediction
    return prediction, local_prediction, spatial_prediction #output shape #(1, 8, 64, 64, 64)


def network_unet(input, num_labels, is_training, data_format='channels_first'):
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = None
    padding = 'reflect'
    with tf.variable_scope('unet'):
        unet = UnetClassicAvgLinear3d(64, 4, data_format=data_format, double_filters_per_level=True, kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        prediction = unet(input, is_training=is_training)
        prediction = conv3d(prediction, num_labels, [1, 1, 1], name='output', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
    return prediction, prediction, prediction


def network_proposed(input, num_labels, is_training, data_format='channels_first'):
    downsampling_factor = 2
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = he_initializer
    spatial_activation = None
    padding = 'reflect'
    with tf.variable_scope('double_net'):
        unet = UnetClassicAvgLinear3d(64, 4, data_format=data_format, double_filters_per_level=True, kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        unet_out = unet(input, is_training=is_training)
        conv1 = conv3d(input, 32, [1, 1, 1], name='conv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        conv2 = conv3d(input, 32, [3, 3, 3], name='conv2', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)
        conv3 = conv3d(input, 32, [5, 5, 5], name='conv3', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)
        concate1 = concat_channels([conv1, conv2, conv3], name='concate1', data_format='channels_first')
        conv4 = conv3d(concate1, 32, [3, 3, 3], name='conv4', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)
        conv5 = concat_channels([conv4, conv3d(conv4, 32, [3, 3, 3], name='conv5', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)], name='add1')
        conv6 = concat_channels([conv5, conv3d(conv5, 32, [3, 3, 3], name='conv6', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)], name='add2')
        concate2 = concat_channels([unet_out, conv6], name='concate2', data_format='channels_first')
        prediction1 = conv3d(concate2, num_labels, [1, 1, 1], name='prediction1', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
    with tf.variable_scope('net2'):
        prediction1_pool = avg_pool3d(prediction1, [downsampling_factor] * 3, name='prediction1_pool')
        conv7 = conv3d(prediction1_pool, 64, [3, 3, 3], name='conv7', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        up = upsample3d_linear(conv7, [downsampling_factor] * 3, name='spatial_prediction', padding='valid_cropped')
        prediction2 = conv3d(up, num_labels, [1, 1, 1], name='prediction2', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)
    with tf.variable_scope('combination'):
        prediction = prediction1 * prediction2
    return prediction, prediction1, prediction2


def network_proposed1(input, num_labels, is_training, data_format='channels_first'):
    downsampling_factor = 4
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = he_initializer
    spatial_activation = None
    padding = 'reflect'
    with tf.variable_scope('double_net'):
        conv1 = conv3d(input, 32, [1, 1, 1], name='conv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        conv2 = conv3d(input, 32, [3, 3, 3], name='conv2', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)
        conv3 = conv3d(input, 32, [5, 5, 5], name='conv3', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)
        concate1 = concat_channels([conv1, conv2, conv3], name='concate1', data_format='channels_first')
        unet = UnetClassicAvgLinear3d(64, 4, data_format=data_format, double_filters_per_level=True,kernel_initializer=kernel_initializer, activation=activation, padding=padding)
        unet_out = unet(concate1, is_training=is_training)
        conv4 = conv3d(concate1, 32, [3, 3, 3], name='conv4', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)
        conv5 = concat_channels([conv4, conv3d(conv4, 32, [5, 5, 5], name='conv5', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)], name='add1')
        conv6 = concat_channels([conv5, conv3d(conv5, 32, [5, 5, 5], name='conv6', padding=padding, kernel_initializer=kernel_initializer,activation=activation, is_training=is_training)], name='add2')
        concate2 = concat_channels([unet_out, conv6], name='concate2', data_format='channels_first')
        prediction1 = conv3d(concate2, num_labels, [1, 1, 1], name='prediction1', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
    with tf.variable_scope('net2'):
        prediction1_pool = avg_pool3d(prediction1, [downsampling_factor] * 3, name='prediction1_pool')
        conv7 = conv3d(prediction1_pool, 64, [3, 3, 3], name='conv7', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        up = upsample3d_linear(conv7, [downsampling_factor] * 3, name='spatial_prediction', padding='valid_cropped')
        prediction2 = conv3d(up, num_labels, [1, 1, 1], name='prediction2', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)
    with tf.variable_scope('combination'):
        prediction = prediction1 * prediction2
    return prediction, prediction1, prediction2


def MCB(inputs, flt, is_training):
    conv1 = conv3d(inputs, flt, [1, 1, 1], name='conv1',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    conv2 = conv3d(inputs, flt, [3, 3, 3], name='conv2',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    conv3 = conv3d(inputs, flt, [5, 5, 5], name='conv3',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    c = concat_channels([conv1, conv2, conv3], name='concate',data_format='channels_first')
    o = conv3d(c, flt, [1, 1, 1], name='conv4',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    return o
def sa(en, de, flt, is_training,name1='',name2='',name3=''):
    h = conv3d(en, flt*4, [1, 1, 1], name =name1,kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    l = conv3d(de, flt*4, [1, 1, 1], name =name2,kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    feature = add([h, l])
    feature = conv3d(feature, flt*4, [1, 1, 1], name = name3,kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect', is_training=is_training)
    out = tf.multiply(h, feature)
    return out
def dul_sa1(en, de, flt,is_training):
    s1 = sa(en, de, flt, is_training, name1='h1',name2='l1', name3='f1')
    s2 = sa(de, en, flt, is_training, name1='h2',name2='l2', name3 = 'f2')
    c = concat_channels([s1, s2],  name='concate_dulsa', data_format='channels_first')
    feature = conv3d(c, 1, [1, 1, 1], name= 'feature',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    feature = tf.layers.batch_normalization(feature)
    feature = tf.nn.relu(feature)
    return feature
def dul_sa2(en, de, flt,is_training):
    s1 = sa(en, de, flt, is_training, name1='h12',name2='l12', name3='f12')
    s2 = sa(de, en, flt, is_training, name1='h22',name2='l22', name3 = 'f22')
    c = concat_channels([s1, s2],  name='concate_dulsa2', data_format='channels_first')
    feature = conv3d(c, 1, [1, 1, 1], name= 'feature2',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    feature = tf.layers.batch_normalization(feature)
    feature = tf.nn.relu(feature)
    return feature
def ca(inputs):
    flt = inputs.get_shape().as_list()[-1]
    avg_pool = GlobalAveragePooling3D()(inputs)
    avg_pool = Reshape((1, 1, 1, flt))(avg_pool)
    max_pool = GlobalMaxPooling3D()(inputs)
    max_pool = Reshape((1, 1, 1, flt))(max_pool)
    feature = Add()([avg_pool, max_pool])
    feature = Activation('sigmoid')(feature)
    return tf.multiply(feature, inputs)
def la(input, is_training):
    i = conv3d(input, 4, [1, 1, 1], name= 'i1',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect',is_training=is_training)
    c = ca(input)
    c = conv3d(c, 16, [1, 1, 1], name= 'cc1',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect',is_training=is_training)
    f1 = conv3d(c, 4, [3, 3, 3], name= 'fw1',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    f2 = conv3d(f1, 4, [1, 1, 1], name= 'fw2',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect',is_training=is_training)
    m = tf.multiply(f1, f2)
    a = add([m, f1])
    out = add([a, i])
    return out
def LA_Net3D(inputs, num_labels, is_training, data_format='channels_first'):
    flt = 16
    with tf.variable_scope('la_net'):
        mcb = MCB(inputs, flt, is_training)
        conv1 = conv3d(mcb, flt, [3, 3, 3], name='convww1',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv2 = conv3d(conv1, flt, [3, 3, 3], name='convww2',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c1 = concat_channels([inputs, conv2], name='concate1', data_format='channels_first')
        pool1 = max_pool3d(c1, [2, 2, 2],  name='pool1', padding='same', data_format='channels_first')

        conv3 = conv3d(pool1, flt*2, [3, 3, 3], name='convww3',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv4 = conv3d(conv3, flt*2, [3, 3, 3], name='convww4',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        enups2 = upsample3d(conv4, kernel_size=[2, 2, 2], name='ct2', data_format='channels_first')
        c2 = concat_channels([pool1, conv4],  name='concate2', data_format='channels_first')
        pool2 = max_pool3d(c2, [2, 2, 2],  name='pool2', padding='same', data_format='channels_first')

        conv5 = conv3d(pool2, flt*4, [3, 3, 3], name='convww5',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv6 = conv3d(conv5, flt*4, [3, 3, 3], name='convww6',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        enups4 = upsample3d(conv6, kernel_size=[4, 4, 4], name='ct4', data_format='channels_first')
        c3 = concat_channels([pool2, conv6],  name='concate3', data_format='channels_first')
        pool3 = max_pool3d(c3, [2, 2, 2],  name='pool3', padding='same', data_format='channels_first')

        conv7 = conv3d(pool3, flt*8, [3, 3, 3], name='convww7',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv8 = concat_channels([conv3d(conv7, flt, [3, 3, 3], name='con71',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same', is_training=is_training), pool3], name='concate31', data_format='channels_first')
        ups8 = upsample3d(conv8, kernel_size=[8, 8, 8], name='ct8', data_format='channels_first')

        up1 = concat_channels([upsample3d(conv8, kernel_size=[2, 2, 2], name='up1', data_format='channels_first'), c3],name='concate4', data_format='channels_first' ) # (60, 40, 22, 1)
        conv9 = conv3d(up1, flt*4, [3, 3, 3], name='convww9',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv10 = conv3d(conv9, flt*4, [3, 3, 3], name='convww10',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c4 = concat_channels([up1, conv10],name='concate5', data_format='channels_first')  # (60, 40, 22, 1)
        deups4 = upsample3d(c4, kernel_size=[4, 4, 4], name='ct44', data_format='channels_first')
        desa4 = dul_sa1(enups4, deups4, flt, is_training)

        up2 = upsample3d(c4, kernel_size=[2, 2, 2], name='ct45', data_format='channels_first')
        up2 = concat_channels([up2, c2] ,name='concate6', data_format='channels_first')  # (120, 80, 44, 1)
        conv11 = conv3d(up2, flt*2, [3, 3, 3], name='convww11',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv12 = conv3d(conv11, flt*2, [3, 3, 3], name='convww12',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c5 = concat_channels([up2, conv12], name='concate61', data_format='channels_first')
        deups2 = upsample3d(c5, kernel_size=[2, 2, 2], name='ct455', data_format='channels_first')
        desa2 = dul_sa2(enups2, deups2, flt, is_training)

        up3 = upsample3d(c5, kernel_size=[2, 2, 2], name='ct4555', data_format='channels_first')
        up3 = concat_channels([up3, c1], name='concate62', data_format='channels_first')  # (240, 160, 88, 1)
        conv13 = conv3d(up3, flt, [3, 3, 3], name='convww13',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv14 = conv3d(conv13, flt, [3, 3, 3], name='convww14',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c6 = concat_channels([up3, conv14], name='concate7', data_format='channels_first')

        c9 = la(concat_channels([ups8, c6, desa2, desa4, ca(mcb)], name='concate8', data_format='channels_first'),is_training)

        output = conv3d(c9, num_labels, [1, 1, 1], name='out',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'same',is_training=is_training)

    return output, output, output


def LA_GUT(input, num_labels, is_training, data_format='channels_first'):
    downsampling_factor = 4
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = he_initializer
    spatial_activation = None
    padding = 'reflect'
    flt=16
    inputs=input
    with tf.variable_scope('la_net'):
        mcb = MCB(inputs, flt, is_training)
        conv1 = conv3d(mcb, flt, [3, 3, 3], name='convww1',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv2 = conv3d(conv1, flt, [3, 3, 3], name='convww2',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c1 = concat_channels([inputs, conv2], name='concate1', data_format='channels_first')
        pool1 = max_pool3d(c1, [2, 2, 2],  name='pool1', padding='same', data_format='channels_first')

        conv3 = conv3d(pool1, flt*2, [3, 3, 3], name='convww3',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv4 = conv3d(conv3, flt*2, [3, 3, 3], name='convww4',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        enups2 = upsample3d(conv4, kernel_size=[2, 2, 2], name='ct2', data_format='channels_first')
        c2 = concat_channels([pool1, conv4],  name='concate2', data_format='channels_first')
        pool2 = max_pool3d(c2, [2, 2, 2],  name='pool2', padding='same', data_format='channels_first')

        conv5 = conv3d(pool2, flt*4, [3, 3, 3], name='convww5',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv6 = conv3d(conv5, flt*4, [3, 3, 3], name='convww6',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        enups4 = upsample3d(conv6, kernel_size=[4, 4, 4], name='ct4', data_format='channels_first')
        c3 = concat_channels([pool2, conv6],  name='concate3', data_format='channels_first')
        pool3 = max_pool3d(c3, [2, 2, 2],  name='pool3', padding='same', data_format='channels_first')

        conv7 = conv3d(pool3, flt*8, [3, 3, 3], name='convww7',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv8 = concat_channels([conv3d(conv7, flt, [3, 3, 3], name='con71',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same', is_training=is_training), pool3], name='concate31', data_format='channels_first')
        ups8 = upsample3d(conv8, kernel_size=[8, 8, 8], name='ct8', data_format='channels_first')

        up1 = concat_channels([upsample3d(conv8, kernel_size=[2, 2, 2], name='up1', data_format='channels_first'), c3],name='concate4', data_format='channels_first' ) # (60, 40, 22, 1)
        conv9 = conv3d(up1, flt*4, [3, 3, 3], name='convww9',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv10 = conv3d(conv9, flt*4, [3, 3, 3], name='convww10',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c4 = concat_channels([up1, conv10],name='concate5', data_format='channels_first')  # (60, 40, 22, 1)
        deups4 = upsample3d(c4, kernel_size=[4, 4, 4], name='ct44', data_format='channels_first')
        desa4 = dul_sa1(enups4, deups4, flt, is_training)

        up2 = upsample3d(c4, kernel_size=[2, 2, 2], name='ct45', data_format='channels_first')
        up2 = concat_channels([up2, c2] ,name='concate6', data_format='channels_first')  # (120, 80, 44, 1)
        conv11 = conv3d(up2, flt*2, [3, 3, 3], name='convww11',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv12 = conv3d(conv11, flt*2, [3, 3, 3], name='convww12',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c5 = concat_channels([up2, conv12], name='concate61', data_format='channels_first')
        deups2 = upsample3d(c5, kernel_size=[2, 2, 2], name='ct455', data_format='channels_first')
        desa2 = dul_sa2(enups2, deups2, flt, is_training)

        up3 = upsample3d(c5, kernel_size=[2, 2, 2], name='ct4555', data_format='channels_first')
        up3 = concat_channels([up3, c1], name='concate62', data_format='channels_first')  # (240, 160, 88, 1)
        conv13 = conv3d(up3, flt, [3, 3, 3], name='convww13',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv14 = conv3d(conv13, flt, [3, 3, 3], name='convww14',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        c6 = concat_channels([up3, conv14], name='concate7', data_format='channels_first')

        c9 = la(concat_channels([ups8, c6, desa2, desa4, ca(mcb)], name='concate8', data_format='channels_first'),is_training)
        local_prediction = conv3d(c9, num_labels, [1, 1, 1], name='local_prediction', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)
        print(local_prediction.shape)#(1, 8, 64, 64, 64)

    with tf.variable_scope('spatial_configuration'):
        local_prediction_pool = avg_pool3d(local_prediction, [downsampling_factor] * 3, name='local_prediction_pool')#[downsampling_factor] * 3 相当于kernel_size [4, 4, 4]
        scconv = conv3d(local_prediction_pool, 64, [5, 5, 5], name='scconv0', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [5, 5, 5], name='scconv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [5, 5, 5], name='scconv2', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        spatial_prediction_pool = conv3d(scconv, num_labels, [5, 5, 5], name='spatial_prediction_pool', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)
        #spatial_prediction_pool: in=[1, 64, 16, 16, 16] out=[1, 8, 16, 16, 16]
        spatial_prediction = upsample3d_linear(spatial_prediction_pool, [downsampling_factor] * 3, name='spatial_prediction', padding='valid_cropped')
    with tf.variable_scope('combination'):
        prediction = local_prediction * spatial_prediction
    return prediction, local_prediction, spatial_prediction #output shape #(1, 8, 64, 64, 64)


