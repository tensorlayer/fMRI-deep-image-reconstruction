from easydict import EasyDict as edict

config = edict()

"data and check point directories"
config.raw_image_dir = '/home/fl4918/FL_deployment/alphatest_imagesGAN_f/train_class'
config.data_tfrecord_dir = '/home/fl4918/FL_deployment/alphaGAN_f/train_class/cifar10_labeled.tfrecord'

# should be 20000 or 50000
config.num_of_data = 5000

"optimization"
config.batch_size = 64
config.lr_init = 2e-4
config.beta1 = 0.5
config.beta2 = 0.9

"loss"
config.loss_type = 'sigmoid'
config.recons_loss_w = 40.0
config.e_adverse_loss_w = 4.0
config.g_gen_loss_w = 4.0

config.n_epoch = 100
config.lr_decay = 0.99

"generator type"
config.generator_type = 'dcgan' # or 'dcgan'

config.use_augmentation = False
"summaries"
config.summary_dir = './summary'

config.save_every = 1
config.num_of_update_for_e_g = 2
config.h_dim = 128
config.num_of_resblk = 0
config.z_dim = 100
config.y_dim = 10