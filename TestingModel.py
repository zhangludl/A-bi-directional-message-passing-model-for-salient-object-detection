import cv2
import numpy as np
import MainModel as MM
import os
import sys
import tensorflow as tf
import time
import vgg16
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def load_img_list(dataset):

    if dataset == 'DUT-OMRON':
        path = '/home/zhanglu/Documents/dataset/dutomron/OMRON-Image'
    elif dataset == 'HKU-IS':
        path = '/home/zhanglu/Documents/dataset/HKU-IS/HKU-IS_Image'
    elif dataset == 'PASCAL-S':
        path = '/home/zhanglu/Documents/dataset/pascal-s/PASCAL-S-'
    elif dataset == 'ECSSD':
        path = '/home/zhanglu/Documents/dataset/ecssd/images/images'
    elif dataset == 'coco':
        path = '/home/zhanglu/Mask_RCNN/val/val'
    elif dataset == 'SED1':
        path = '/home/zhanglu/Documents/dataset/SED1/SED1-Image'
    elif dataset == 'SED2':
        path = '/home/zhanglu/Documents/dataset/SED2/SED2-Image'
    elif dataset == 'SOC':
        path = '/home/zhanglu/Downloads/SOC6K_Release/ValSet/img_select'
    elif dataset == 'zy':
        path = '/home/zhanglu/Documents/zengyi_1981_1024'


    imgs = os.listdir(path)

    return path, imgs


if __name__ == "__main__":

    model = MM.Model()
    
    model.build_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = MM.img_size
    label_size = MM.label_size
    ckpt = tf.train.get_checkpoint_state('model')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)
    datasets = ['zy']
    if not os.path.exists('Result'):
        os.mkdir('Result')

    for dataset in datasets:
        path, imgs = load_img_list(dataset)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result/' + dataset + '/map'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            img_name, ext = os.path.splitext(f_img)
            
            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))

                start_time = time.time()
                sal_map,result = sess.run([model.Score,model.Prob],
                                  feed_dict={model.input_holder: img})
      
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join(save_dir, img_name+'.jpg')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

                
    sess.close()
