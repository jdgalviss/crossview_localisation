from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputData
import tensorflow as tf
import numpy as np
import os
import cv2

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
NETWORK_TYPE = 'CVM-NET-I'

SATELLITE_IMAGE_PREFIX = '../Data/CVUSA/test_sat/'
GROUND_IMAGE_PREFIX = '../Data/CVUSA/dubai2/ground'
NUM_SAMPLES = 8
# -------------------------------------------------------- #

class CVMInference:
    def __init__(self):
        self.sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
        self.grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_count = 0
        self.aerial_image_prefix = SATELLITE_IMAGE_PREFIX
        self.ground_image_prefix = GROUND_IMAGE_PREFIX
        self.num_imgs = NUM_SAMPLES

        # build model
        if NETWORK_TYPE == 'CVM-NET-I':
            self.sat_global, self.grd_global = cvm_net_I(self.sat_x, self.grd_x, self.keep_prob, False)
        elif NETWORK_TYPE == 'CVM-NET-II':
            self.sat_global, self.grd_global = cvm_net_II(self.sat_x, self.grd_x, self.keep_prob, False)
        else:
            print ('CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid')
        
        # run model
        print('CVMInference object created')
        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            print('loading model...')
            load_model_path = '../Model/' + NETWORK_TYPE + '/' + str(0) + '/model.ckpt'
            self.saver.restore(sess, load_model_path)
            print("   Model loaded from: %s" % load_model_path)
            print('load model...FINISHED')
            print('Test with black images')
            sat_zeros = np.zeros([1,512,512, 3])
            grd_zeros = np.zeros([1,224,1232, 3])
            feed_dict = {self.sat_x: sat_zeros, self.grd_x: grd_zeros, self.keep_prob: 1.0}
            sat_global_val, grd_global_val = sess.run([self.sat_global, self.grd_global], feed_dict=feed_dict)
            print('Test on black images passed')
    

    def load_images(self, is_ground = True, num_images = 5):
        if(is_ground):
            images = np.zeros([num_images, 224, 1232, 3], dtype = np.float32)
            for i in range(num_images):
                img = cv2.imread(self.ground_image_prefix + str(i) + '.jpg')
                img = cv2.resize(img, (1232, 224), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32)
                # img -= 100.0
                img[:, :, 0] -= 103.939  # Blue
                img[:, :, 1] -= 116.779  # Green
                img[:, :, 2] -= 123.6  # Red
                images[i, :, :, :] = img
            return images
        else:
            images = np.zeros([num_images, 512, 512, 3], dtype=np.float32)
            for i in range(num_images):
                img = cv2.imread(SATELLITE_IMAGE_PREFIX + str(i) + '.jpg')
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32)
                # img -= 100.0
                img[:, :, 0] -= 103.939  # Blue
                img[:, :, 1] -= 116.779  # Green
                img[:, :, 2] -= 123.6  # Red
                images[i, :, :, :] = img
            return images
    
    def next_images(self, is_ground = True, batch_size = 1):
        if(self.global_count >= self.num_imgs):
            return None
        else:
            if(is_ground):
                images = np.zeros([batch_size, 224, 1232, 3], dtype = np.float32)
                for i in range(batch_size):
                    img = cv2.imread(self.ground_image_prefix + str(self.global_count) + '.jpg')
                    img = cv2.resize(img, (1232, 224), interpolation=cv2.INTER_AREA)
                    self.global_count += 1
                    img = img.astype(np.float32)
                    # img -= 100.0
                    img[:, :, 0] -= 103.939  # Blue
                    img[:, :, 1] -= 116.779  # Green
                    img[:, :, 2] -= 123.6  # Red
                    images[i, :, :, :] = img
                return images
            else:
                images = np.zeros([batch_size, 512, 512, 3], dtype=np.float32)
                for i in range(batch_size):
                    img = cv2.imread(self.aerial_image_prefix + str(self.global_count) + '.jpg')
                    self.global_count += 1
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32)
                    # img -= 100.0
                    img[:, :, 0] -= 103.939  # Blue
                    img[:, :, 1] -= 116.779  # Green
                    img[:, :, 2] -= 123.6  # Red
                    images[i, :, :, :] = img
                return images
        
        
        
    
    def load_images_raw(self, is_ground = True, num_images = 5):
        if(is_ground):
            images = np.zeros([num_images, 224, 1232, 3], dtype = np.uint8)
            for i in range(num_images):
                #print(SATELLITE_IMAGE_PREFIX + str(i) + '.jpg')
                img = cv2.imread(self.ground_image_prefix + str(i) + '.jpg')
                img = cv2.resize(img, (1232, 224), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                images[i, :, :, :] = img
            return images
        else:
            images = np.zeros([num_images, 512, 512, 3], dtype=np.uint8)
            for i in range(num_images):
                img = cv2.imread(self.aerial_image_prefix + str(i) + '.jpg')
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                images[i, :, :, :] = img
            return images
            
            
    def preprocess(self, images, is_ground):
        num_images = images.shape[0]
        images = images.astype(np.float32)
        for image in images:
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red

            image[:, :, 0] -= 103.939  # Blue
            image[:, :, 1] -= 116.779  # Green
            image[:, :, 2] -= 123.6  # Red
        return images
    
    def forward(self, images, is_ground = True):
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            load_model_path = '../Model/' + NETWORK_TYPE + '/' + str(0) + '/model.ckpt'
            self.saver.restore(sess, load_model_path)
            sat_zeros = np.zeros([1,512,512, 3])
            grd_zeros = np.zeros([1,224,1232, 3])
            # ---------------------- validation ----------------------
            print('computing global descriptors...')
            sat_zeros = np.zeros([images.shape[0],512,512, 3])
            grd_zeros = np.zeros([images.shape[0],224,1232, 3])
            
            if(is_ground):
                feed_dict = {self.sat_x: sat_zeros, self.grd_x: images, self.keep_prob: 1.0}
            else:
                feed_dict = {self.sat_x: images, self.grd_x: grd_zeros, self.keep_prob: 1.0}
                
            sat_global_val, grd_global_val = sess.run([self.sat_global, self.grd_global], feed_dict=feed_dict)

            if(is_ground):
                global_descriptor = grd_global_val
            else:
                global_descriptor = sat_global_val
                
            return global_descriptor
