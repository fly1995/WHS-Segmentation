import tensorflow as tf
from tensorflow_train.layers.layers import conv3d, concat_channels, avg_pool3d, conv3d_transpose, add,max_pool3d, upsample3d,dense,dropout
from tensorflow_train.layers.interpolation import upsample3d_linear
from tensorflow_train.networks.unet_base import UnetBase
from tensorflow_train.layers.initializers import he_initializer
from tensorflow_train.layers.normalizers import batch_norm
from tensorflow.python.keras.layers import Activation, Concatenate,BatchNormalization, Conv3DTranspose, Add, Dropout,Dense,Embedding
from keras.optimizers import Adam
from keras.layers import Lambda, Reshape, Input, concatenate, Conv3D,Dropout,BatchNormalization, add, Concatenate, Activation,GlobalAveragePooling3D, Reshape,Add,multiply,GlobalMaxPooling3D
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
import tensorflow as tf
from Transformer import  *


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
    i = conv3d(input, 4, [1, 1, 1], name= 'i11',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect',is_training=is_training)
    c = ca(input)
    c = conv3d(c, 16, [1, 1, 1], name= 'cc1',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect',is_training=is_training)
    f1 = conv3d(c, 4, [3, 3, 3], name= 'fw1',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'reflect',is_training=is_training)
    f2 = conv3d(f1, 4, [1, 1, 1], name= 'fw2',kernel_initializer=he_initializer, activation=tf.nn.sigmoid, padding= 'reflect',is_training=is_training)
    m = tf.multiply(f1, f2)
    a = add([m, f1])
    out = add([a, i])
    return out



def mlp(x):
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    return x
def conv_block(inputs, num_filters, is_training):
    x = conv3d(inputs, num_filters, [3, 3, 3],kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
    x = conv3d(x, num_filters, [3, 3, 3],kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
    return x
def decoder_block(inputs, skip_features, num_filters,is_traning):
    x = upsample3d(inputs, kernel_size=[2, 2, 2], name='', data_format='channels_first')
    x = tf.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters,is_traning)
    return x
def transformer_encoder(x):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(8)(x)
    x = Add()([x, skip_1])
    skip_2 = x
    y = Lambda (lambda e: LayerNormalization()(e))(x)
    y1 = mlp(y)
    y2 = Add()([y1, skip_2])
    return y2



def LA_GUT_MHD(input, num_labels, is_training, data_format='channels_first'):
    downsampling_factor = 4
    kernel_initializer = he_initializer
    activation = tf.nn.relu
    local_kernel_initializer = he_initializer
    local_activation = tf.nn.tanh
    spatial_kernel_initializer = he_initializer
    spatial_activation = None
    padding = 'reflect'
    flt=24
    inputs=input
    with tf.variable_scope('la_net'):
        # mcb = MCB(inputs, flt, is_training)
        conv1 = conv3d(inputs, flt, [3, 3, 3], name='convww1',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
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

        patch_embed = conv3d(pool3, 256, [4, 4, 4], name='patch_embd', kernel_initializer=he_initializer, activation=tf.nn.relu, padding='same', is_training=is_training)
        _, f, h, w, d = patch_embed.shape  #(1, 96, 8, 8, 8)
        patch_embed = Lambda(lambda o: Reshape((h * w * d, f))(o))(patch_embed)  # (?,512,96)
        # position embedings
        positions = tf.range(start=0, limit=512, delta=1)
        pos_embed = Embedding(input_dim=512, output_dim=256)(positions)
        embed = patch_embed+pos_embed
        x = embed
        for _ in range(12):
            x1 = transformer_encoder(x)
        _, _, xdim, ydim, zdim = pool3.shape
        pool3 = Lambda(lambda o: Reshape((f, 8, 8, 8))(o))(x1)

        conv7 = conv3d(pool3, flt*8, [3, 3, 3], name='convww7',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same',is_training=is_training)
        conv8 = concat_channels([conv3d(conv7, flt, [3, 3, 3], name='con71',kernel_initializer=he_initializer, activation=tf.nn.relu, padding= 'same', is_training=is_training), pool3], name='concate31', data_format='channels_first')
        # ups8 = upsample3d(conv8, kernel_size=[8, 8, 8], name='ct8', data_format='channels_first')

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

        c9 = concat_channels([c6, desa2, desa4], name='concate8', data_format='channels_first')
        conv15 = conv3d(c9, flt, [3, 3, 3], name='convww15', kernel_initializer=he_initializer,activation=tf.nn.relu, padding='same', is_training=is_training)
        c9 = la(concat_channels([conv15, desa2, desa4], name='LA', data_format='channels_first'),is_training)
        # c91 = la(c90, is_training=is_training)
        local_prediction = conv3d(c9, num_labels, [1, 1, 1], name='local_prediction', padding=padding, kernel_initializer=local_kernel_initializer, activation=local_activation, is_training=is_training)

    with tf.variable_scope('spatial_configuration', reuse=tf.AUTO_REUSE):
        local_prediction_pool = avg_pool3d(local_prediction, [downsampling_factor] * 3, name='local_prediction_pool')#[downsampling_factor] * 3 相当于kernel_size [4, 4, 4]
        scconv = conv3d(local_prediction_pool, 64, [3, 3, 3], name='scconv0', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [3, 3, 3], name='scconv1', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        scconv = conv3d(scconv, 64, [3, 3, 3], name='scconv2', padding=padding, kernel_initializer=kernel_initializer, activation=activation, is_training=is_training)
        spatial_prediction_up = upsample3d_linear(scconv, [downsampling_factor] * 3,name='spatial_prediction_up', padding='valid_cropped')
        spatial_prediction = conv3d(spatial_prediction_up, num_labels, [1, 1, 1], name='spatial_prediction', padding=padding, kernel_initializer=spatial_kernel_initializer, activation=spatial_activation, is_training=is_training)

    with tf.variable_scope('combination', reuse=tf.AUTO_REUSE):
        prediction = local_prediction * spatial_prediction
    return prediction, local_prediction, spatial_prediction #output shape #(1, 8, 64, 64, 64)

