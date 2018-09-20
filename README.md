# fMRI-deep-image-reconstruction
## Image Generation (Conditional AlphaGAN)
This is a Tensorflow / Tensorlayer implementation of α-GAN, combined with conditional GAN for generating images to be used in EEG & fMRI deep image reconstruction.

α-GAN: [Variational Approaches for Auto-Encoding Generative Adversarial Networks](https://arxiv.org/abs/1706.04987)
Conditional GAN: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

Tensorflow - v1.8.0
Tensorlayer - v1.9.0

### Usage
The training dataset must first be converted into a `.tfrecord` format.  
This can be done by going to `utils.py` and modifying `class_text_to_int(label)` to contain the list of classes, and running `convert_tfrecord(data_dir, save_dir, filename)`.  An example is provided at the bottom of `utils.py` which you can run by executing `utils.py`.
*(`data_dir` should contain all the folders with the dataset labels, and all the dataset images should be in their respective folder)*

Before training the Conditional α-GAN, make sure the directory paths in `config.py` correspond to the dataset locations.

Execute the training by running the following command
```
python3 main.py
```