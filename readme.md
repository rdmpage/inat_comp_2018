# iNaturalist Competition 2018 Training Code  
This code fine tunes an Inception V3 model (pretrained on ImageNet) on the iNaturalist 2018 competition [dataset](https://github.com/visipedia/inat_comp).

### Background

This repository https://github.com/rdmpage/inat_comp_2018 is a fork of https://github.com/juancprzs/inat_comp_2018, which itself is a fork of the original repository https://github.com/macaodha/inat_comp_2018. I have updated the original README to reflect my own experience getting this code to work on an Apple MacBook Pro with an M1 chip.

The original code includes the ability to format the output for the [iNaturalist Challenge at FGVC5 on Kaggle](https://www.kaggle.com/competitions/inaturalist-2018/data).

#### Apple Mac version

A few tweaks are required to get this code to work. I am using  Python 3.11.3 on an Apple MacBook Pro running macOS Ventura 13.4. Some of the changes are because the libraries used by the original code have changed, others are because I’m running on a Mac. These all seem straightforward, but some involved a fair amount of hair pulling and teeth gnashing (partly reflecting that I am a newbie to both Python and machine learning).

- change `workers` from 10 to 8, and `batch_size` from 64 to 32.- set data source and training/validation/test splits to my files
- change type `np.int` to `int`
- replace call to `view` function with `replace`
- use Apple Metal (`mps` device)
- remove `.module` keys from `state_dict` so model can be saved and read correctly

Once running, for larger data sets the program would quit complaining: `OSError: [Errno 24] Too many open files`. This can be fixed by finding out the system limit for open files (`ulimit -n`), which on my Mac was 256, then increasing that limit, e.g.: `ulimit -n 1024`


### Training
~~The network was trained on Ubuntu 16.04 using PyTorch 0.3.0. Each training epoch took about 1.5 hours using a GTX Titan X~~.  
The links for the raw data are available [here](https://github.com/visipedia/inat_comp).
~~We also provide a trained model that can be downloaded from [here](http://vision.caltech.edu/~macaodha/inat2018/iNat_2018_InceptionV3.pth.tar).~~

Every epoch the code will save a checkpoint and the current best model according to validation accuracy.  
~~Training for 75 epochs results in a top one accuracy of 60.20% and top three of 77.91% on the validation set~~.


### Ideas for Improvement  
* Train/test on higher resolution images.  
* Make use of the taxonomy at training time (already included in data loader).  
* Address long tail distribution.


### Submission File
By setting the following flags it's possible to generate a submission file for the competition.
```python
    evaluate = True
    save_preds = True
    resume = 'model_path/iNat_2018_InceptionV3.pth.tar'  # path to trained model
    val_file = 'ann_path/test2018.json'                  # path to test file
    data_root = 'data_path/inat2018/images/'             # path to test images
    op_file_name = 'inat2018_test_preds.csv'             # submission filename
```

Note that when running in this model the validation scores will be zero as the test file is just a list of images. Unlike the validation file it lacks annotations saying what each image is. This is, of course, the point of the challenge, participants are asked to submit identifications for a set of unclassified images.

### Reading

Van Horn G, Mac Aodha O, Song Y, Cui Y, Sun C, Shepard A, Adam H, Perona P, Belongie S (2017) The iNaturalist Species Classification and Detection Dataset. In: arXiv.org. https://arxiv.org/abs/1707.06642v2. Accessed 10 Jul 2023

Horn GV, Aodha OM, Song Y, Cui Y, Sun C, Shepard A, Adam H, Perona P, Belongie S (2018) The iNaturalist Species Classification and Detection Dataset. IEEE Computer Society, pp 8769–8778 https://doi.org/10.1109/CVPR.2018.00914

