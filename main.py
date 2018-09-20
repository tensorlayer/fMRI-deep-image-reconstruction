from models import *
from config import *
from utils import *
import glob
import time
import numpy as np
import argparse
from scipy import misc
import pickle
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--image_raw_data_dir', type=str, default=config.raw_image_dir,
                    help='directory contains all raw images with size (224, 224, 3)')

parser.add_argument('--mode', type=str, default='train', help='train or evaluate')

"optimizer"
batch_size = config.batch_size
lr_init = config.lr_init
beta1 = config.beta1
beta2 = config.beta2

"loss"
loss_type = config.loss_type

"adversarial learning (GAN)"
n_epoch = config.n_epoch
lr_decay = config.lr_decay

"tfrecord data file"
filename = config.data_tfrecord_dir

"summaries"
summary_dir = config.summary_dir

num_of_update_for_e_g = config.num_of_update_for_e_g

recons_loss_w = config.recons_loss_w

e_adverse_loss_w = config.e_adverse_loss_w

g_gen_loss_w = config.g_gen_loss_w

save_every_epoch = config.save_every

num_of_resblk = config.num_of_resblk

h_dim = config.h_dim

z_dim = config.z_dim

y_dim = config.y_dim

g_type = config.generator_type

is_augment = config.use_augmentation

num_of_data = config.num_of_data


def train():

    # Modify path to change test image folder
    test_images = get_test_images(dir='/home/fl4918/FL_deployment/alphaGAN_f/train_samples')

    t_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_image')
    t_label = tf.placeholder(tf.float32, [None, y_dim], name='label')
    net_e, z_hat = encoder((t_image/127.5)-1, resblk=num_of_resblk, z_dim=z_dim, is_train=True, reuse=False)
    t_z = tf.placeholder(tf.float32, [None, z_dim], name='z_prior')

    net_g, x_gen = generator(z=t_z, y=t_label, h_dim=h_dim, is_train=True, reuse=False)

    _, x_recons = generator(z=z_hat, y=t_label, h_dim=h_dim, is_train=True, reuse=True)

    net_cd, cd_logits_fake = code_discriminator(z_hat, reuse=False)

    _, cd_logits_real = code_discriminator(t_z, reuse=True)

    net_d, d_logits_fake1 = discriminator(x_recons, is_train=True, reuse=False)

    _, d_logits_fake2 = discriminator(x_gen, is_train=True, reuse=True)

    _, d_logits_real = discriminator((t_image/127.5)-1, is_train=True, reuse=True)

    "define test network"
    net_e_test, z_test = encoder((t_image/127.5)-1, resblk=num_of_resblk, z_dim=z_dim, is_train=False, reuse=True)

    net_g_test, _ = generator(z=z_test, y=t_label, h_dim=h_dim, is_train=False, reuse=True)

    "define another test network to evaluate the generative performance of generator"
    net_g_test1, _ = generator(z=t_z, y=t_label, h_dim=h_dim, is_train=False, reuse=True)

    np.random.seed(42)
    sampled_z_test = np.random.normal(0.0, 1.0, [64, z_dim])

    targets = []

    for i in range(64):
        targets.append(i//8)
    sampled_y_test = np.eye(y_dim)[targets]

    "auto encoder loss"
    reconstruction_loss = recons_loss_w*tf.reduce_mean(tf.losses.absolute_difference(
        x_recons, (t_image/127.5)-1
    ))

    with tf.name_scope('s_encoder'):
        if loss_type == 'lse':
            e_loss1 = e_adverse_loss_w*tf.reduce_mean(tf.squared_difference(cd_logits_fake,
                                                                            tf.ones_like(cd_logits_fake)))
        else:
            e_loss1 = e_adverse_loss_w*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=cd_logits_fake, labels=tf.ones_like(cd_logits_fake)))

        e_loss = e_loss1 + reconstruction_loss

        "define summaries"
        s_e_recons_loss = tf.summary.scalar('reconstruction_loss',
                                            reconstruction_loss)
        s_e_adverse_loss = tf.summary.scalar('adverse_loss', e_loss1)
        s_e_overall_loss = tf.summary.scalar('overall_loss', e_loss)
        e_merge = tf.summary.merge([s_e_recons_loss, s_e_adverse_loss, s_e_overall_loss])
        e_summary_writer = tf.summary.FileWriter(summary_dir+'/encoder')

    with tf.name_scope('s_generator'):

        "generator loss"
        if loss_type == 'lse':
            g_loss1 = tf.reduce_mean(tf.squared_difference(d_logits_fake1,
                                                           tf.ones_like(d_logits_fake1)))
        elif loss_type == 'sigmoid':
            g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake1, labels=tf.ones_like(d_logits_fake1)))
        if loss_type == 'lse':
            g_loss2 = g_gen_loss_w*tf.reduce_mean(tf.squared_difference(d_logits_fake2,
                                                                        tf.ones_like(d_logits_fake2)))
        elif loss_type == 'sigmoid':
            g_loss2 = g_gen_loss_w*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake2, labels=tf.ones_like(d_logits_fake2)))

        g_loss = reconstruction_loss + g_loss1 + g_loss2

        "define summaries"
        s_g_adverse_recons_loss = tf.summary.scalar('adverse_recons_loss', g_loss1)
        s_g_adverse_gen_loss = tf.summary.scalar('adverse_gen_loss', g_loss2)
        s_g_reconstruction_loss = tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        s_g_overall_loss = tf.summary.scalar('overall_loss', g_loss)

        g_merge = tf.summary.merge([s_g_adverse_gen_loss, s_g_adverse_recons_loss,
                                    s_g_reconstruction_loss, s_g_overall_loss])

        g_summary_writer = tf.summary.FileWriter(summary_dir+'/generator')

    with tf.name_scope('s_discriminator'):
        "discriminator loss"
        if loss_type == 'lse':
            d_loss1 = tf.reduce_mean(tf.squared_difference(d_logits_fake1,
                                                           tf.zeros_like(d_logits_fake1)))
        elif loss_type == 'sigmoid':
            d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake1,
                                                                             labels=tf.zeros_like(d_logits_fake1)))

        if loss_type == 'lse':
            d_loss2 = tf.reduce_mean(tf.squared_difference(d_logits_fake2,
                                                           tf.zeros_like(d_logits_fake2)))
        elif loss_type == 'sigmoid':
            d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake2,
                                                                             labels=tf.zeros_like(d_logits_fake2)))

        if loss_type == 'lse':
            d_loss3 = tf.reduce_mean(tf.squared_difference(d_logits_real,
                                                           tf.ones_like(d_logits_real)))
        elif loss_type == 'sigmoid':
            d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                             labels=tf.ones_like(d_logits_real)))

        d_loss = d_loss1 + d_loss2 + d_loss3

        "define summaries"
        s_d_adverse_recons_loss = tf.summary.scalar('adverse_recons_loss', d_loss1)
        s_d_adverse_gen_loss = tf.summary.scalar('adverse_gen_loss', d_loss2)
        s_d_real_loss = tf.summary.scalar('adverse_real_loss', d_loss3)
        s_d_overall_loss = tf.summary.scalar('overall_loss', d_loss)

        d_merge = tf.summary.merge([s_d_adverse_gen_loss, s_d_adverse_recons_loss, s_d_real_loss, s_d_overall_loss])
        d_summary_writer = tf.summary.FileWriter(summary_dir+'/discriminator')

    with tf.name_scope("s_code_discriminator"):
        "code discriminator loss"
        if loss_type == 'lse':
            cd_loss1 = tf.reduce_mean(tf.squared_difference(cd_logits_fake,
                                                            tf.zeros_like(cd_logits_fake)))
        else:
            cd_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_fake,
                                                                              labels=tf.zeros_like(cd_logits_fake)))

        if loss_type == 'lse':
            cd_loss2 = tf.reduce_mean(tf.squared_difference(cd_logits_real,
                                                            tf.ones_like(cd_logits_real)))
        else:
            cd_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_real,
                                                                              labels=tf.ones_like(cd_logits_real)))
        cd_loss = cd_loss1 + cd_loss2

        s_cd_adverse_loss_fake = tf.summary.scalar('adverse_loss_fake', cd_loss1)
        s_cd_adverse_loss_real = tf.summary.scalar('adverse_loss_real', cd_loss2)
        s_cd_overall_loss = tf.summary.scalar('overall_loss', cd_loss)

        cd_merge = tf.summary.merge([s_cd_adverse_loss_fake, s_cd_adverse_loss_real, s_cd_overall_loss])
        cd_summary_writer = tf.summary.FileWriter(summary_dir+'/code_discriminator')

    e_vars = tl.layers.get_variables_with_name(name='encoder', train_only=True, printable=True)
    g_vars = tl.layers.get_variables_with_name(name='generator', train_only=True, printable=True)
    d_vars = tl.layers.get_variables_with_name(name='discriminator', train_only=True, printable=True)
    cd_vars = tl.layers.get_variables_with_name(name='code_evaluator', train_only=True, printable=True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(e_loss, var_list=e_vars)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)
    cd_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(cd_loss, var_list=cd_vars)

    save_gan_dir = "./samples/train_gan"
    save_test_gan_dir = "./samples/test_gan"
    checkpoints_dir = "./checkpoints"

    tl.files.exists_or_mkdir(save_gan_dir)
    tl.files.exists_or_mkdir(checkpoints_dir)
    tl.files.exists_or_mkdir(save_test_gan_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    tl.layers.initialize_global_variables(sess)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoints_dir+"/g_{}.npz".format(tl.global_flag['mode']),
                                 network=net_g)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoints_dir+"/e_{}.npz".format(tl.global_flag['mode']),
                                 network=net_e)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoints_dir+"/cd_{}.npz".format(tl.global_flag['mode']),
                                 network=net_cd)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoints_dir+"/d_{}.npz".format(tl.global_flag['mode']),
                                 network=net_d)

    num_of_iter_one_epoch = num_of_data // batch_size

    sess.run(tf.assign(lr_v, lr_init))
    print("Training alpha-GAN with initialized learning rate: %f" % lr_init)

    img_batch, label_batch = input_batch(filename, batch_size, n_epoch, shuffle_size=50000, is_augment=is_augment)

    try:
        epoch_time = time.time()
        n_iter = 0
        while True:
            if (n_iter + 1) % num_of_iter_one_epoch == 0:
                log = "[*] Epoch [%4d/%4d] time: %4.4fs" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch, time.time()-epoch_time
                )
                print(log)
                lr_new = lr_init * (lr_decay**((n_iter+1)//num_of_iter_one_epoch))
                print((lr_decay**((n_iter+1)//num_of_iter_one_epoch)))
                sess.run(tf.assign(lr_v, lr_new))
                print("Training alpha-GAN with new learning rate: %f" % lr_new)
                epoch_time = time.time()

            step_time = time.time()

            imgs, labels = sess.run([img_batch, label_batch])

            imgs = np.array(imgs)

            labels_one_hot = np.eye(y_dim)[labels-1]

            batch_sz = imgs.shape[0]
            "sample a standard normal distribution"
            z_prior = np.random.normal(0, 1.0, (batch_sz, z_dim))

            "update encoder and generator multiple times"
            for i in range(num_of_update_for_e_g):
                "update encoder"
                e_summary, err_E_recons_loss, err_E_adversarial_loss, err_E_loss, _ = sess.run(
                    [e_merge, reconstruction_loss, e_loss1, e_loss, e_optim],
                    feed_dict={t_image: imgs, t_z: z_prior, t_label: labels_one_hot})

                log = "Epoch [%4d/%4d] %6d time: %4.4fs, e_loss: %8f, e_recons_loss: %8f, e_adverse_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch, n_iter, time.time() - step_time, err_E_loss,
                    err_E_recons_loss, err_E_adversarial_loss
                )

                print(log)

                e_summary_writer.add_summary(e_summary, n_iter*num_of_iter_one_epoch + i)

                "update generator"
                g_summary, err_G_recons_loss, err_G_adverse_loss, err_G_gen_loss, err_G_loss, _ = sess.run(
                    [g_merge, reconstruction_loss, g_loss1, g_loss2, g_loss, g_optim],
                    feed_dict={t_image: imgs, t_z: z_prior, t_label: labels_one_hot}
                )

                log = "Epoch [%4d/%4d] %6d time: %4.4fs, g_loss: %8f, g_recons_loss: %8f, g_adverse_loss: %8f, g_gen_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch, n_iter, time.time() - step_time, err_G_loss, err_G_recons_loss,
                    err_G_adverse_loss, err_G_gen_loss
                )

                print(log)

                g_summary_writer.add_summary(g_summary, n_iter*num_of_iter_one_epoch + i)

            "update discriminator"
            d_summary, err_D_real_loss, err_D_recons_loss, err_D_gen_loss, err_D_loss, _ = \
                sess.run([d_merge, d_loss3, d_loss1, d_loss2, d_loss, d_optim],
                         feed_dict={t_image: imgs, t_z: z_prior, t_label: labels_one_hot})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, d_loss: %8f, d_recons_loss: %8f, d_gen_loss: %8f, d_real_loss: %8f" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_D_loss, err_D_recons_loss,
                err_D_gen_loss, err_D_real_loss
            )
            print(log)

            d_summary_writer.add_summary(d_summary, n_iter)

            "update code discriminator"

            cd_summary, err_CD_fake_loss, err_CD_real_loss, err_CD_loss, _ = \
                sess.run([cd_merge, cd_loss1, cd_loss2, cd_loss, cd_optim],
                         feed_dict={t_image: imgs, t_z: z_prior, t_label: labels_one_hot})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, cd_loss: %8f, cd_fake_loss: %8f, cd_real_loss: %8f" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch, n_iter, time.time() - step_time, err_CD_loss, err_CD_fake_loss,
                err_CD_real_loss
            )

            print(log)

            cd_summary_writer.add_summary(cd_summary, n_iter)

            if (n_iter+1) % (save_every_epoch * num_of_iter_one_epoch) == 0:
                tl.files.save_npz(net_g.all_params,
                                  name=checkpoints_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_e.all_params,
                                  name=checkpoints_dir + '/e_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_d.all_params,
                                  name=checkpoints_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_cd.all_params,
                                  name=checkpoints_dir + '/cd_{}.npz'.format(tl.global_flag['mode']), sess=sess)

            if (n_iter + 1) % (num_of_iter_one_epoch * save_every_epoch) == 0:
                # quick evaluation on train set
                t_label_test = np.zeros((batch_sz, y_dim))
                out = sess.run(net_g_test.outputs,
                               {t_image: test_images, t_label: t_label_test})
                out = (out+1)*127.5
                print("reconstructed image:", out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out.astype(np.uint8), [8, 8], save_gan_dir +
                                   '/train_%d.png' % ((n_iter + 1) // num_of_iter_one_epoch))

                # quick evaluation on generative performance of generator
                out1 = sess.run(net_g_test1.outputs, feed_dict={t_z: sampled_z_test, t_label: sampled_y_test})
                out1 = (out1+1)*127.5
                print("generated image:", out1.shape, out1.min(), out1.max())
                print("[*] save images")
                tl.vis.save_images(out1.astype(np.uint8), [8, 8], save_test_gan_dir
                                   + '/test_%d.png' % ((n_iter + 1) // num_of_iter_one_epoch))

            n_iter += 1

    except tf.errors.OutOfRangeError:
        print("Training Done")
        pass


def generate():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    result_dir = './results/recons'
    checkpoints_dir = "./checkpoints"
    tl.files.exists_or_mkdir(result_dir)

    with open('./results/encoded_feat.pkl', 'rb') as f:
        feat_dict = pickle.load(f)

    encoded_feats = feat_dict['encoded_feats']
    image_ids = feat_dict['image_ids']

    # error?
    predict_label = feat_dict['pred_label']

    t_z = tf.placeholder(tf.float32, [None, z_dim], name='test_prior')
    t_pred_label = tf.placeholder(tf.float32, [None, y_dim], name='labels')
    net_g_test, _ = generator(z=t_z, y=t_pred_label, is_train=False, reuse=False)

    tl.layers.initialize_global_variables(sess)

    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoints_dir+"/g_train.npz",
                                 network=net_g_test)
    for i in range(len(image_ids)):
        cur_feat = np.reshape(encoded_feats[i, :], (-1, z_dim))
        cur_pred_label = np.reshape(predict_label[i, :], (-1, y_dim))
        out = sess.run(net_g_test.outputs, feed_dict={t_z: cur_feat, t_pred_label: cur_pred_label})
        out = (out+1)*127.5
        out = np.reshape(out, (64, 64, 3))
        misc.imsave(result_dir+'/'+image_ids[i], out.astype(np.uint8))


def encode():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    result_dir = './results'
    checkpoints_dir = "./checkpoints"
    tl.files.exists_or_mkdir(result_dir)
    t_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name='Input_images')
    net_e_test, z_test = encoder((t_image/127.5)-1, resblk=num_of_resblk,
                                 z_dim=z_dim, is_train=False, reuse=False)
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoints_dir+"/e_train.npz", network=net_e_test)

    encoded_feats = None

    # Make sure directory is correct
    images_dir = '/home/fl4918/FL_deployment/alphaGAN_f/train_samples'

    image_paths = glob.glob(images_dir+'/*')

    image_ids = []
    images = []

    for img_path in image_paths:
        image_id = img_path.split('/')[-1]
        im = misc.imread(img_path)
        im = misc.imresize(im, [64, 64])

        image_ids.append(image_id)

        images.append(im)
    images = np.array(images)
    num_of_sample = images.shape[0]

    num_of_batches = (num_of_sample // batch_size) + 1

    for i in range(num_of_batches):
        start_idx = i*batch_size
        end_idx = (i+1)*batch_size

        cur_batch = images[start_idx:end_idx, :, :, :]
        cur_encoded_feat = sess.run(z_test, feed_dict={t_image: cur_batch})
        if encoded_feats is None:
            encoded_feats = cur_encoded_feat
        else:
            encoded_feats = np.vstack((encoded_feats, cur_encoded_feat))

    with open(result_dir + '/encoded_feat.pkl', 'wb') as f:
        pickle.dump({'image_ids': image_ids, 'encoded_feats': encoded_feats}, f)


if __name__ == "__main__":
    with tf.device('/cpu:0'):
        args = parser.parse_args()

        tl.global_flag['mode'] = args.mode

        if args.mode == 'train':
            train()

        elif args.mode == 'encode':
            encode()

        elif args.mode == 'gen':
            generate()
        elif args.mode == 'generate':
            generate()

        else:
            raise Exception('Unknown mode {}'.format(args.mode))

