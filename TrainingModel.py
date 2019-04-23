import cv2
import numpy as np
import vgg16
import MainModel as MM
import tensorflow as tf
import os
import argparse

def load_training_list():

    # 'train_list.txt' is the list of image names of the training dataset.
    with open('train_list.txt') as f:
        lines = f.read().splitlines()

    files = []
    labels = []
    for line in lines:
        labels.append('data/label/%s' % line.replace('.jpg', '.png'))#path of dataset
        files.append('data/image/%s' % line)
    return files, labels


def train(lr,n_epochs,save_dir,clip_grads = None, load = None, model_files = None):

    opt = tf.train.AdamOptimizer(lr)
    with tf.variable_scope(tf.get_variable_scope()):

        model = MM.Model()
        model.build_model()
        tvars = tf.trainable_variables()
        grads = tf.gradients(model.Loss_Mean, tvars)
        if clip_grads:
            max_grad_norm = 1
            clip_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    train_op = opt.apply_gradients(zip(grads, tvars))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #
    if load:
        ckpt = tf.train.get_checkpoint_state(model_files)
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_list, label_list= load_training_list()

    img_size = MM.img_size
    label_size = MM.label_size

    for i in range(1,n_epochs):
        whole_loss = 0.0
        whole_acc = 0.0
        count = 0

        for f_img, f_label in zip(train_list, label_list):

            img = cv2.imread(f_img).astype(np.float32)
            img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
            img = img.reshape((1, img_size, img_size, 3))
            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            label = cv2.resize(label, (label_size, label_size))
            label = label.astype(np.float32) # the input GT has been preprocessed to [0,1]
            label = np.stack((label, 1-label), axis=2)
            label = np.reshape(label, [-1, 2])
            _, loss, acc = sess.run([train_op, model.Loss_Mean, model.accuracy],
                                    feed_dict={model.input_holder: img,
                                               model.label_holder: label
                                               })
            whole_loss += loss
            whole_acc += acc
            count = count + 1
            if count % 200 == 0:
                print "Loss of %d images: %f, Accuracy: %f" % (count, (whole_loss/count), (whole_acc/count))
        save_dir = save_dir + '/model.ckpt'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print "Epoch %d: %f" % (i, (whole_loss/len(train_list)))
        saver.save(sess, save_dir, global_step=i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default='0',type = str) # gpu id
    parser.add_argument('-e', type = int) # epochs
    parser.add_argument('-l', type = float) # learning rate
    parser.add_argument('-c', default = False, action = 'store_true') # whether to use grads clip
    parser.add_argument('-a', default = False, action = 'store_true') # whether to load a pretrained model
    parser.add_argument('-m', default=None, type = str) # path to pretrained model
    parser.add_argument('-s', type = str) # path to save ckpt file


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    train(lr = args.l,
          model_files=args.m,
          n_epochs=args.e,
          save_dir=args.s,
          clip_grads=args.c,
          load=args.a)