import tensorflow as tf
import vgg16
import cv2
import numpy as np

img_size = 256
label_size = img_size
fea_dim = 128
class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder = tf.placeholder(tf.float32, [label_size*label_size, 2])
        
    
    def build_model(self):
        # gbd
        vgg = self.vgg
        vgg.build(self.input_holder)

        
        conv5_dilation = self.dilation(vgg.conv5_3, 512, 32, 'conv5')
        conv4_dilation = self.dilation(vgg.conv4_3, 512, 32, 'conv4')
        conv3_dilation = self.dilation(vgg.conv3_3, 256, 32, 'conv3')
        conv2_dilation = self.dilation(vgg.conv2_2, 128, 32, 'conv2')
        conv1_dilation = self.dilation(vgg.conv1_2, 64, 32, 'conv1')
        with tf.variable_scope('fusion') as scope:
            h0_1 = conv5_dilation
            h0_3 = conv4_dilation
            h0_5 = conv3_dilation
            h0_7 = conv2_dilation
            h0_9 = conv1_dilation
            
            h1_1 = tf.nn.relu(self.Conv_2d(h0_1, [3, 3, 128, 128], 0.01, name='h1_1'))
         
            h1_3 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h1_1, [3, 3, 128, 128], 0.01, name='h1_1_3')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_1, [3, 3, 128, 128], 0.01, name='g1_1_3')),
                                          [32, 32]) + \
                   tf.nn.relu(self.Conv_2d(h0_3, [3, 3, 128, 128], 0.01, name='h1_3'))
            
            h1_5 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h1_3, [3, 3, 128, 128], 0.01, name='h1_3_5')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_3, [3, 3, 128, 128], 0.01, name='g1_3_5')),
                                          [64, 64]) + \
                   tf.nn.relu(self.Conv_2d(h0_5, [3, 3, 128, 128], 0.01, name='h1_5'))
            
            h1_7 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h1_5, [3, 3, 128, 128], 0.01, name='h1_5_7')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_5, [3, 3, 128, 128], 0.01, name='g1_5_7')),
                                          [128, 128]) + \
                   tf.nn.relu(self.Conv_2d(h0_7, [3, 3, 128, 128], 0.01, name='h1_7'))
            
            h1_9 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h1_7, [3, 3, 128, 128], 0.01, name='h1_7_9')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_7, [3, 3, 128, 128], 0.01, name='g1_7_9')),
                                          [256, 256]) + \
                   tf.nn.relu(self.Conv_2d(h0_9, [3, 3, 128, 128], 0.01, name='h1_9'))
            ## 
            h2_9 = tf.nn.relu(self.Conv_2d(h0_9, [3, 3, 128, 128], 0.01, name='h2_9'))
            
            h2_7 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h2_9, [3, 3, 128, 128], 0.01, name='h2_9_7')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_9, [3, 3, 128, 128], 0.01, name='g2_9_7')),
                                          [128, 128]) + \
                   tf.nn.relu(self.Conv_2d(h0_7, [3, 3, 128, 128], 0.01, name='h2_7'))
            
            h2_5 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h2_7, [3, 3, 128, 128], 0.01, name='h2_7_5')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_7, [3, 3, 128, 128], 0.01, name='g2_7_5')),
                                          [64, 64]) + \
                   tf.nn.relu(self.Conv_2d(h0_5, [3, 3, 128, 128], 0.01, name='h2_5'))
            
            h2_3 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h2_5, [3, 3, 128, 128], 0.01, name='h2_5_3')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_5, [3, 3, 128, 128], 0.01, name='g2_5_3')),
                                          [32, 32]) + \
                   tf.nn.relu(self.Conv_2d(h0_3, [3, 3, 128, 128], 0.01, name='h2_3'))
            
            h2_1 = tf.image.resize_images(tf.nn.relu(self.Conv_2d(h2_3, [3, 3, 128, 128], 0.01, name='h2_3_1')) *
                                          tf.nn.sigmoid(self.Conv_2d(h0_3, [3, 3, 128, 128], 0.01, name='g2_3_1')),
                                          [16, 16]) + \
                   tf.nn.relu(self.Conv_2d(h0_1, [3, 3, 128, 128], 0.01, name='h2_1'))
            ## 
            h3_1 = tf.nn.relu(
                self.Conv_2d(tf.concat([h1_1, h2_1], axis=3), [3, 3, 256, 128], 0.01, name='h3_1'))
            h3_3 = tf.nn.relu(
                self.Conv_2d(tf.concat([h1_3, h2_3], axis=3), [3, 3, 256, 128], 0.01, name='h3_3'))
            h3_5 = tf.nn.relu(
                self.Conv_2d(tf.concat([h1_5, h2_5], axis=3), [3, 3, 256, 128], 0.01, name='h3_5'))
            h3_7 = tf.nn.relu(
                self.Conv_2d(tf.concat([h1_7, h2_7], axis=3), [3, 3, 256, 128], 0.01, name='h3_7'))
            h3_9 = tf.nn.relu(
                self.Conv_2d(tf.concat([h1_9, h2_9], axis=3), [3, 3, 256, 128], 0.01, name='h3_9'))
            prev5 = tf.nn.relu(self.Conv_2d(h3_1 , [3, 3, 128, 64], 0.01, name='prev5_1'))
            prev5 = self.Conv_2d(prev5, [1, 1, 64, 2], 0.01, padding='VALID', name='prev5')
            prev5 = tf.image.resize_images(prev5, [32, 32])
            prev4 = tf.nn.relu(self.Conv_2d(h3_3, [3, 3, 128, 64], 0.01, name='prev4_1'))
            prev4 = self.Conv_2d(prev4, [1, 1, 64, 2], 0.01, padding='VALID', name='prev4') + prev5
            prev4 = tf.image.resize_images(prev4, [64, 64])
            prev3 = tf.nn.relu(self.Conv_2d(h3_5, [3, 3, 128, 64], 0.01, name='prev3_1'))
            prev3 = self.Conv_2d(prev3, [1, 1, 64, 2], 0.01, padding='VALID', name='prev3') + prev4
            prev3 = tf.image.resize_images(prev3, [128, 128])
            prev2 = tf.nn.relu(self.Conv_2d(h3_7 , [3, 3, 128, 64], 0.01, name='prev2_1'))
            prev2 = self.Conv_2d(prev2, [1, 1, 64, 2], 0.01, padding='VALID', name='prev2') + prev3
            prev2 = tf.image.resize_images(prev2, [256, 256])
            prev1 = tf.nn.relu(self.Conv_2d(h3_9 , [3, 3, 128, 64], 0.01, name='prev1_1'))
            prev1 = self.Conv_2d(prev1, [1, 1, 64, 2], 0.01, padding='VALID', name='prev1') + prev2

            


        self.Score = tf.reshape(prev1, [-1, 2])
        #
        self.Prob = tf.nn.softmax(self.Score)
       
        self.Loss_Mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.Score, labels=self.label_holder))
        self.correct_prediction = tf.equal(tf.argmax(self.Score, 1), tf.argmax(self.label_holder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



    
    def dilation(self,input_,input_dim,output_dim,name):
        with tf.variable_scope(name) as scope:
            a = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 1, 0.01, name = "dilation1"))
            b = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 3, 0.01, name ='dilation3'))
            c = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 5, 0.01, name = 'dilation5'))
            d = tf.nn.relu(self.Atrous_conv2d(input_, [3, 3, input_dim, output_dim], 7, 0.01, name = 'dilation7'))
            e = tf.concat([a,b,c,d],axis = 3)
        return e
    
    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            # b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            b = tf.get_variable('b', shape=[shape[3]],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)

            return conv

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME', name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv
    def Atrous_conv2d(self,input_,shape,rate,stddev,name,padding = 'SAME'):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                            shape = shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
            atrous_conv = tf.nn.atrous_conv2d(input_,W,rate = rate,padding=padding)
            b = tf.get_variable('b', shape=[shape[3]], initializer=tf.constant_initializer(0.0))
            atrous_conv = tf.nn.bias_add(atrous_conv, b)
        return atrous_conv

    
