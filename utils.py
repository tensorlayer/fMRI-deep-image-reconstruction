from random import shuffle
import scipy.misc as misc
import numpy as np
import glob
import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def imread(path):
    return misc.imread(path).astype(np.float)


def get_image(image_path, image_size, is_crop=True, resize_w=64):
    return transform(imread(image_path), image_size, is_crop, resize_w)


def get_test_images(dir='./train_samples'):
    img_paths = glob.glob(dir+'/*')
    imgs = []
    for img_path in img_paths:
        im = misc.imread(img_path)
        im = misc.imresize(im, [64, 64])
        imgs.append(im)

    return np.array(imgs)


def convert_tfrecord(data_dir='./train_class', save_dir='./train_class', filename='converted'):
    tl.files.exists_or_mkdir(save_dir)
    filedir = os.path.join(save_dir, filename + '.tfrecord')
    imgdir = glob.glob(os.path.join(data_dir + '/*/*.jpg'))

    print('Directories Set')

    #for filen in glob.glob(os.path.join('/home/fl4918/FL_deployment/alphaGAN_f/train_class/*/*.jpg')):
        #print(filen)

    shuffle(imgdir)

    labels = list()
    for path in imgdir:
        temp = os.path.dirname(path)
        temp = os.path.basename(temp)
        # print(temp)

        labels.append(temp)

    writer = tf.python_io.TFRecordWriter(filedir)

    for i in range(len(imgdir)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(imgdir)))

        # load image
        img = load_image(imgdir[i])
        label = labels[i]

        # print(class_text_to_int(label))
        feature = {'label': _int64_feature(class_text_to_int(label)),
                   'image_raw': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def load_image(imgdir):
    img = misc.imread(imgdir)
    img = misc.imresize(img, [64, 64])
    img = img.astype(np.float32)
    return img


# for cifar 10, modify based on your classes
def class_text_to_int(label):
    if label == 'automobile':
        return 1
    elif label == 'bird':
        return 2
    elif label == 'cat':
        return 3
    elif label == 'deer':
        return 4
    elif label == 'dog':
        return 5
    elif label == 'frog':
        return 6
    elif label == 'horse':
        return 7
    elif label == 'ship':
        return 8
    elif label == 'truck':
        return 9
    elif label == 'airplane':
        return 10
    else:
        return 0

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def input_batch(filename, batch_size, num_epochs, shuffle_size, is_augment):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode)

        if is_augment:
            dataset = dataset.map(augment)

        dataset = dataset.shuffle(buffer_size=shuffle_size)

        dataset = dataset.repeat(num_epochs)

        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.float32)

    label = tf.cast(features['label'], tf.int32)

    image.set_shape((64*64*3))

    image = tf.reshape(image, (64, 64, 3))

    return image, label


def augment(img, label):
    "j"
    image_size_r = int(64*1.2)
    "1. randomly flip the image from left to right"
    img = tf.image.random_flip_left_right(img)

    "2. rotate the image counterclockwise 90 degree"
    img = tf.image.rot90(img, k=1)

    img = tf.image.random_flip_up_down(img)
    img = tf.image.resize_images(img, size=[image_size_r, image_size_r], method=tf.image.ResizeMethod.BICUBIC)

    img = tf.random_crop(img, [64, 64, 3])

    return img, label


if __name__ == "__main__":
    convert_tfrecord('/home/fl4918/FL_deployment/alphaGAN_f/train_class/',
                     '/home/fl4918/FL_deployment/alphaGAN_f/train_class/', 'cifar10_labeled')

