# fMRI-deep-image-reconstruction
## Image Generation (Alpha-GAN)
This is a Tensorflow / Tensorlayer implementation of α-GAN for generating images to be used in EEG & fMRI deep image reconstruction.

α-GAN: [Variational Approaches for Auto-Encoding Generative Adversarial Networks](https://arxiv.org/abs/1706.04987)

Tensorflow - v1.8.0

Tensorlayer - v1.9.0

### Usage
#### Training
The training dataset must first be converted into a `.tfrecord` format.  

This can be done by going to `utils.py` and modifying `class_text_to_int(label)` to contain the list of classes, and running `convert_tfrecord(data_dir, save_dir, filename)`.  An example is provided at the bottom of `utils.py` which you can run by executing `utils.py`.

*(`data_dir` should contain all the folders with the dataset labels, and all the dataset images should be in their respective folder)*

Before training the α-GAN, make sure the directory paths in `config.py` correspond to the dataset locations.

Execute the training by running the following command
```
python3 main.py
```
This will train the α-GAN and save the model in `checkpoints_dir` every epoch.

Generator testing is split into two parts: training set, and generation performance.  These two are saved in `save_gan_dir` and `save_test_gan_dir` respectively.

#### Encoding
This extracts the features from the given folder of images using the trained encoder, and stores them in `encoded_feat.pkl`.

```
python3 main.py --mode=encode
```

#### Generating
This reconstructs the folder of images from the encoding section by using the extracted features from `encoded_feat.pkl` to generate images.

```
python3 main.py --mode=gen
python3 main.py --mode=generate
```

