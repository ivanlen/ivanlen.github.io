---
title: "How to generate synthetics Mars' surface images using StyleGAN"
layout: post
date: 2020-01-30
# image: /assets/images/markdown.jpg
headerImage: false
tag:
- DNNs
- Mars
- images
- Neural Networks
- Generative Adversarial Networks
star: true
category: blog
# hidden: true # don't count this post in blog pagination
projects: true
author: ivanlen
description: Markdown summary with different options
---

## How to generate synthetics Mars' surface images using StyleGAN

If we are asked to draw a picture of the surface of Mars, we will probably draw a reddish surface, perhaps with a crater or some other geographical feature. Although the surface may be more complex, and have various colors and distinct shapes, minerals and characteristics, we agree that we can characterize the properties of the surface. If we were good enough drawers and knew the properties of the surface of Mars well enough, we could generate fake images that would even fool a specialist, since the properties of the images would be the same as a real image and the quality of the image would be the same.

Generative Adversarial Networks (GANs) are a neural networks that are composed by two networks: a Generator network -the artist- and a Discriminator network -the specialist- which are trained in a two player game. The generator tries to fool the discriminator by generating real looking images, and the discriminator tries to distinguish between real and fake making the generator to do its best to fool the discriminator. If we iterate over this game with enough fake and real examples, the generator will learn how to synthesize fake images that have the same statistics or properties as the real images.

In this tutorial we will see how to train a GAN developed by Nvidia, the [StyleGAN](https://arxiv.org/abs/1812.04948), to exploit this idea and generate synthetic images of Mars' surface which look like real ones. The aim of this tutorial is to show hot to train end-to-end a GAN to generate good quality synthetic images and discuss some things of the pipeline. I will assume that you know some basic concepts about machine learning and some basic python stuff.


I strongly recommend to read some theory or material about generative models and GANs because it is fascinating and revolutional, in fact Yann [LeCun described GANs](https://www.youtube.com/watch?v=IbjF5VjniVE) as *the coolest idea in machine learning in the last twenty years*.

### Teaser

#### Fake or real?
Synthetic images

<a href="/assets/posts/mars_tutorial/fakes012084.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/fakes012084.jpg" width="99%" /></a> <!-- markup clean_ -->

Real images

<a href="/assets/posts/mars_tutorial/reals_mosaic.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/reals_mosaic.jpg" width="99%" /></a> <!-- markup clean_ -->

Can you tell the difference?

<a href="/assets/posts/mars_tutorial/test_512_3.gif" target="_blank">Training evolution (it may take a while to load ~18Mb)</a> <!-- markup clean_ -->



### Some useful theoretical resources

- A ver nice [tutorial](https://poloclub.github.io/ganlab/){:target="_ blank"} to play with and get some intuition about these concepts
- Ian's Goodfellow -the creator of GANs- [tutorial](https://arxiv.org/abs/1701.00160){:target="_ blank"}
- Stanford [Lecture](https://www.youtube.com/watch?v=5WoItGTWV54){:target="_ blank"} about generative models

Now, let's get started!

#### Requirements

- Linux, MacOs or Windows are supported, but I strongly recommend using a Linux machine.
- One o more GPUs with that support CUDA at least 8Gb or RAM.
- 64-bit python (that we are going to install using anaconda)


## Generating the training set

### Obtaining the raw images

As in every machine learning pipeline, we need a training dataset to train our GAN. We will use images provided by [HiRISE](https://medium.com/r/?url=https%3A%2F%2Fwww.uahirise.org%2Fcatalog%2F){:target="_ blank"} (High Resolution Imaging Science Experiment), which is the the most powerful camera ever sent to another planet. In the catalog of HiRISE there are lots of images that look like these

<div id="cover">
  <a href="/assets/posts/mars_tutorial/samples/27002215442_179da57f4b_o.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/samples/27002215442_179da57f4b_o.jpg" width="32%" /></a> <!-- markup clean_ -->
   <a href="/assets/posts/mars_tutorial/samples/esp_040663_1415.jpg"><img src="/assets/posts/mars_tutorial/samples/esp_040663_1415.jpg" width="32%" /></a>
  <a href="/assets/posts/mars_tutorial/samples/esp_044675_2580.jpg"><img src="/assets/posts/mars_tutorial/samples/esp_044675_2580.jpg" width="32%" /></a>

  <a href="/assets/posts/mars_tutorial/samples/pia21609.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/samples/pia21609.jpg" width="32%" /></a> <!-- markup clean_ -->
  <a href="/assets/posts/mars_tutorial/samples/pia22513.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/samples/pia22513.jpg" width="32%" /></a> <!-- markup clean_ -->
  <a href="/assets/posts/mars_tutorial/samples/PIA22587.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/samples/PIA22587.jpg" width="32%" /></a> <!-- markup clean_ -->
</div>


 <!-- To download the images go to the [catalog](https://medium.com/r/?url=https%3A%2F%2Fwww.uahirise.org%2Fcatalog%2F),
 <img src="/assets/posts/mars_tutorial/example-1.png" width="50%" class="image-float-left" alt=""> open an image that you like and then for example [this one](https://www.uahirise.org/ESP_062586_1840), and download the IRB-color non-map image. These are jpg images. -->

 To download the images go to the [catalog](https://medium.com/r/?url=https%3A%2F%2Fwww.uahirise.org%2Fcatalog%2F){:target="_ blank"},  open an image that you like and then for example [this one](https://www.uahirise.org/ESP_062586_1840){:target="_ blank"}, and download the IRB-color non-map image. These are jpg images.

There are different time of images, some of them are wallpapers, some others are very tall ones, and also there are different sizes.

Download as many as you can and for the moment do not worry about the different shapes of the images. Then we are going to process the images so they all have the same shape. In this tutorial the images that we are going to generate are of 512x512 px size, so the only consideration that you have to take care is that the smaller dimension of the images must be at least 512px.

So, to have a raw dataset I downloaded 344 jpg images of the database and drop them all in the same folder, in my case is the `/datasets/raw` folder.

```bash
.
└── datasets
    └── raw
```

```bash
.
├── datasets
│   ├── cropped_files_aug
│   ├── cropped_files_no_aug
│   ├── no_borders
│   └── raw
├── stylegan
│   ├── datasets
│   ├── dnnlib
│   ├── metrics
│   └── training
└── stylegan_env
    ├── bin
    ├── compiler_compat
    ├── conda-meta
    ├── include
    ├── lib
    ├── share
    ├── ssl
    └── x86_64-conda_cos6-linux-gnu
```

### Creating a python environment (move before data creating the dataset)!

We will install several packages to be able to train the GAN and some of them need specific versions and we don't want to interact or change the version of other packages. Thus, to keep on we need to setup the python virtual environment.
> A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated spaces for them that contain per-project dependencies for them.

Basically using a _virtualenv_ we can install a lots of packages and even broke the environment without damaging other python installations or changing the versions of other packages in other environments.
I strongly recommend using [(ana)conda](https://docs.anaconda.com/anaconda/){:target="_ blank"}, since it is very easy to install python and setup a [virtual environment with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html){:target="_ blank"}. In particular I suggest installing the miniconda version which has the minimal components, and is quite faster.
Follow [these steps](https://docs.anaconda.com/anaconda/install/){:target="_ blank"} to install anaconda on your OS or [these steps](https://docs.conda.io/en/latest/miniconda.html){:target="_ blank"} to install miniconda.
If everything worked find you will be able to use the conda command in your terminal with either of the two installations.

```bash
$ conda V                                                                                                             
conda 4.5.11
```
Next we create a python environment using conda. You will be asked to accept and install some python packages.

```bash
$ conda create --prefix ./stylegan_env python=3.7
```
After the creation of the environment you will find a new folder.
```bash
.
└── datasets
│   └── raw
└── stylegan_env
    ├── ...
    ...
```    

To activate the environment

```bash
~/stylegan_tutorial $ conda activate ./stylegan_env
(/home/ivan/stylegan_tutorial/stylegan_env)
~/stylegan_tutorial $
```

(To deactivate the environment the command is `conda deactivate`.)

### Installing dependencies

Having the virtualenv activated, we will install the dependencies needed to train the GAN.

```bash
$ conda install six pillow requests
```

The implementation that we are going to use of StyleGAN was released for the version 1.10.0 or higher (still 1.x) of tensorflow. Nowadays tensorflow already has released the 2.x version that has some changes that will make the tutorial a little bit harder to follow. Thus, we are going to install the last 1.x version.

```bash
$ conda install tensorflow-gpu==1.14.0
```

Now that we have the environment ready we are ready to run some code.

### Preparing the dataset

### Converting images to RGB

If by any change you download a couple of images that are in black and white or have the wrong RGB format, we are going to standardize the dimension of all the images.

{: .prompt}
In [1]:

```python
import os
from PIL import Image
import numpy as np
```

{: .prompt}
In [2]:

```python
image_dir = './datasets/raw/'
files = os.listdir(image_dir)

for fname in files[:]:
    im = Image.open(image_dir + '/' + fname)
    imarr = np.asarray(im)
    if imarr.ndim==2:
        print(fname)
        rgbimg = Image.new("RGB", im.size)
        rgbimg.paste(im)
        rgbimg.save(image_dir + fname)
```

#### Removing the borders

You might find lots of images that have a red border like this one. Although this border might represent less than the 0.1% of the image, I don't want to include this part of the image since it may introduce noise to the dataset.
<img src="/assets/posts/mars_tutorial/red_borders.png" width="50%" class="image-center" alt="Red borders">
So I will move all 133 the images that contain borders to `'./raw/with_borders/'`. Then I will run this simple script that you can run either on jupyter or as a python script:

{: .prompt}
In [1]:

```python
from PIL import Image
import os
```

{: .prompt}
In [2]:

```python
folder_to_scan = './datasets/raw/with_borders/'
files = os.listdir(folder_to_scan)
saving_folder_name = './datasets/curated/'
valid_file_types = ['.jpg']
```

{: .prompt}
In [3]:

```python
len(files)
```

{: .prompt}
Out [3]:




    133



{: .prompt}
In [4]:

```python
valid_files = []
for file in files:
    f, fe = os.path.splitext(file)
    if fe in valid_file_types:
        valid_files.append(file)

full_path_valid_files = [folder_to_scan + file for file in valid_files]
for file_path in full_path_valid_files:

    file_name = os.path.basename(file_path)
    base_name, file_ext = os.path.splitext(file_name)
    im = Image.open(file_path)

    im_w = im.size[0]
    im_h = im.size[1]
    # delete superior border

    imc = im.crop((0,40,im_w,im_h)) #left bottom right upper

    f_name = saving_folder_name + base_name + file_ext
    imc.save(f_name)

```
After this procedure the raw images will no longer have the upper border.

Finally we can now add the images that have no border to `'./datasets/curated/'`.
My whole raw dataset consists of 343 images, and their sizes range from `512px, 10459px` to `1328px, 1184px`

```bash
.
└── datasets
│   ├── curated
│   └── raw
│       └── no_border
└── stylegan_env
    ├── ...
    ...
```    

#### More preprocessing - data augmentation <a name="dataaugmentation"></a>

To be able to use the images that we have, first we need to reshape the images into a square shape; in general GANs use square images.
For this particular tutorial we are going to cut the raw images into images of size `512px x 512 px`.
This size will generate high quality images and the training process is not going to be SOOOO long.
The bigger the images sizes, the larger the dataset that you will need to train the GAN.
As a starting point I suggest trying `128px x 128px`, with this size you will train considerable faster the GAN compared to `512px x 512 px`.
I used a NVIDIA Titan V, and I took me 2 days of training for the `128px x 128px` and around 3 weeks for the `512px x 512 px` images. Yes, I know, this sounds like a lot of time, but training GANs takes a lot of time.
**It is important to note that whichever size you choose it must be a multiple of 2.**
Additionally we are also to make some data augmentation.


So, now we are going to cut into `512px x 512 px` pieces all the raw images.
And we define some functions that we are going to use also for the data augmentation.
> *Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data.*

> *Training deep learning neural network models on more data can result in more skillful models, and the augmentation techniques can create variations of the images that can improve the ability of the fit models to generalize what they have learned to new images.*


In this particular case we are going to rotate and flip the images in different directions.
For satellite images this procedure will work well because we expect that the properties of the images that we want the GAN to learn, do not change when the image is rotated. For example, a crater will still be a crater even though we rotate or flip the images.
In other cases, for example faces or landscapes, we have to be more careful because we don't want to generate upside down faces or portraits.
*Data augmentation, or image augmentation, has to keep the properties of the images unchanged under the transformations that we make to the images*.


{: .prompt}
In [1]:

```python
from PIL import Image
import os
import itertools
```

{: .prompt}
In [2]:

```python
from PIL.ImageOps import flip, mirror
import PIL
```

{: .prompt}
In [3]:

```python
folder_to_scan = './datasets/curated/'
files = os.listdir(folder_to_scan)
len(files)
```

{: .prompt}
Out [3]:




    343



{: .prompt}
In [4]:

```python
valid_file_types = ['.jpg']
valid_files = []
for file in files:
    f, fe = os.path.splitext(file)
    if fe in valid_file_types:
        valid_files.append(file)
full_path_valid_files = [folder_to_scan+file for file in valid_files]
```

{: .prompt}
In [7]:

```python
def generate_1d_limits(wind, limit, thresh):
    x_left = []
    x_right = []
    if limit >= wind:
        x_lim_reached = False
        i = 0
        while not x_lim_reached:
            x_l = i * wind
            x_r = (i + 1) * wind

            if x_r <= limit:
                x_right.append(x_r)
                x_left.append(x_l)
            else:
                x_lim_reached = True
                # some extra padding
                if (x_r - limit) / wind < thresh:
                    x_r = limit
                    x_l = limit - wind
                    x_right.append(x_r)
                    x_left.append(x_l)
            i += 1
    return (x_left, x_right)


def generate_cropping_boxes_from_limits(x_left, x_rigth, x_bottom, x_top):
    croping_boxes = []
    x_lims = [(x_l, x_r) for x_l, x_r in zip(x_left, x_rigth)]
    y_lims = [(x_l, x_r) for x_l, x_r in zip(x_bottom, x_top)]
    bounding_boxes = list(itertools.product(x_lims, y_lims))
    for i in range(len(bounding_boxes)):
        ((x1, x2), (y1, y2)) = bounding_boxes[i]
        croping_boxes.append((x1, y1, x2, y2))
    return croping_boxes


def generate_cropping_boxes(image, cropping_window, tresh):
    image_width, image_height = image.size
    x_left, x_rigth = generate_1d_limits(cropping_window, image_width, tresh)
    x_bottom, x_top = generate_1d_limits(cropping_window, image_height, tresh)
    croping_boxes = generate_cropping_boxes_from_limits(x_left, x_rigth, x_bottom, x_top)
    return croping_boxes


def image_square_resize(im_input, new_size):
    im = im_input.copy()
    im = im.resize((new_size, new_size), PIL.Image.ANTIALIAS)    
    return im


def image_rotator(im_input, angle):
    if angle==90:
        return im_input.transpose(Image.ROTATE_90)
    elif angle == 180:
        return im_input.transpose(Image.ROTATE_180)
    elif angle == 270:
        return im_input.transpose(Image.ROTATE_270)
    else:
        raise ValueError('angle not supported')    


def image_augmentator(im_input, return_orig = True):
    im_aug = []
    if return_orig:
        im_aug.append(im_input.copy())
    # 1.flip
    im_aug.append(flip(im_input.copy()))
    # 2. rot 180
    im_aug.append(image_rotator(im_input.copy(), 180))
    # 3. flip(rot_90)
    im_aug.append(flip(image_rotator(im_input.copy(), 90)))
    # 4. flip(rot_270)
    im_aug.append(flip(image_rotator(im_input.copy(), 270)))
    return im_aug        
```

In case you want to list the raw images sizes that you are going to use uncomment this code snippet:

{: .prompt}
In [8]:

```python
# class SizeCounter:
#     def __init__(self):
#         self.counter = {}

#     def add(self, size):
#         try:
#             self.counter[size]
#             self.counter[size] +=1
#         except KeyError:
#             self.counter[size] = 1

# sizes = SizeCounter()
# for file in full_path_valid_files:
#     im = Image.open(file)
#     sizes.add(size=im.size)

# # sizes.counter
```

##### No augmentation

{: .prompt}
In [10]:

```python
cropping_window = 512
saving_folder_name = './datasets/cropped_files_no_aug/'
padding_tresh = 0.25
resize = False
image_output_size = 512

os.makedirs(saving_folder_name, exist_ok=True)
```

{: .prompt}
In [11]:

```python
for file in full_path_valid_files:
    im = Image.open(file)
    croping_boxes = generate_cropping_boxes(im, cropping_window, padding_tresh)
    file_name = os.path.basename(file)
    base_name, file_ext = os.path.splitext(file_name)
    for i,b in enumerate(croping_boxes):
        imc = im.crop(b) #left bottom right upper
        if resize:
            imc = image_square_resize(imc,image_output_size )
        f_name = saving_folder_name + base_name +'_{}_'.format(i) + file_ext
        imc.save(f_name)
print(len(os.listdir(saving_folder_name)))
```

{: .prompt}
Out [11]:

    4120


##### Augmentation

{: .prompt}
In [19]:

```python
cropping_window = 512
saving_folder_name = './datasets/cropped_files_aug/'
padding_tresh = 0.25
resize = False
image_output_size = 512

os.makedirs(saving_folder_name, exist_ok=True)
```

{: .prompt}
In [12]:

```python
for file in full_path_valid_files:    
    im = Image.open(file)
    for ia, im in enumerate(image_augmentator(im)):
        croping_boxes = generate_cropping_boxes(im, cropping_window, padding_tresh)
        file_name = os.path.basename(file)
        base_name, file_ext = os.path.splitext(file_name)
        for i,b in enumerate(croping_boxes):
            imc = im.crop(b) #left bottom right upper
            if resize:
                imc = image_square_resize(imc,image_output_size )
            f_name = saving_folder_name + base_name +'_{}_{}_'.format(ia,i) + file_ext
            imc.save(f_name)

print(len(os.listdir(saving_folder_name)))
```

{: .prompt}
Out [12]:

    20600

So first if we only cut the images into `(512, 512)` images we obtain `4120` images.
After the augmentation we in which we keep the original + 4 transformations we have `5 * 4120 = 20600` images. For this dataset this number is enough.

```bash
.
└── datasets
│   ├── curated
│   ├── cropped_files_aug
│   ├── cropped_files_no_aug
│   └── raw
│       └── no_border
└── stylegan_env
    ├── ...
    ...
```    

For this particular case I wrote my own augmentation scripts because I don't need a huge increase in the data set, and just some simple transformations are enough.
But if you want to see which kind of techniques are use for image augmentation you can checkout this links.

#### Some more data augmentation resources:
- https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9
- https://keras.io/preprocessing/image/
- https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/


These are a couple of snapshots of some of the images after the augmentation procedure.

<a href="/assets/posts/mars_tutorial/reals_mosaic.jpg" target="_blank"><img src="/assets/posts/mars_tutorial/reals_mosaic.jpg" width="99%" class="image-center" alt="Red borders"></a> <!-- markup clean_ -->

So for the moment we have around 20k `512, 512 pxs` images ready to be used to train the GAN.

In the next steps we are going to implement the StyleGAN, so before we move on I strongly recommend reading the [StyleGAN README](https://github.com/NVlabs/stylegan/blob/master/README.md){:target="_ blank"}.
You don't need to understand everything but you will gain some intuition at least.

### Clone the StyleGAN repository

So, now we need to clone the [StyleGAN repository](https://github.com/NVlabs/stylegan){:target="_ blank"} where you will find the tensorflow implementation of the StyleGAN.
One option is to [download directly](https://github.com/NVlabs/stylegan/archive/master.zip){:target="_ blank"} the repo from github, the other one is to clone it using [git](https://gist.github.com/derhuerst/1b15ff4652a867391f03){:target="_ blank"}

Using git:
```bash
git clone https://github.com/NVlabs/stylegan.git
```

```bash
.
└── datasets
│   ├── curated
│   ├── cropped_files_aug
│   ├── cropped_files_no_aug
│   └── raw
│       └── no_border
├── stylegan
└── stylegan_env
    ├── ...
    ...
```    

<!-- ### Creating a python environment (move before data creating the dataset)!

To keep on we need to setup the python virtual environment.
> A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated spaces for them that contain per-project dependencies for them.

Basically now we can install a lots of packages and even broke the environment without damaging other python installations or changing the versions of other packages.
I strongly recommend using [(ana)conda](https://docs.anaconda.com/anaconda/), since it is very easy to install python and setup a [virtual environment with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html). In particular I suggest installing the miniconda version which has the minimal components, and is quite faster.
Follow [these steps](https://docs.anaconda.com/anaconda/install/) to install anaconda on your OS or [these steps](https://docs.conda.io/en/latest/miniconda.html) to install miniconda.
If everything worked find you will be able to use the conda command in your terminal with either of the two installations.

```bash
$ conda V                                                                                                             
conda 4.5.11
```
Next we create a python environment using conda. You will be asked to accept and install some python packages.

```bash
$ conda create --prefix ./stylegan_env python=3.7
```
After the creation of the environment you will find a new folder.
```bash
$ ls
datasets  stylegan  stylegan_env
```
Now we activate the environment

```bash
~/stylegan_tutorial $ conda activate ./stylegan_env
(/home/ivan/stylegan_tutorial/stylegan_env)
~/stylegan_tutorial $
```

(To deactivate the environment the command is `conda deactivate`.) -->


### Create tensorflow records

Next we need to create _tfrecords_ from our data set. From the StyleGAN github:
> The training and evaluation scripts operate on datasets stored as multi-resolution TFRecords. Each dataset is represented by a directory containing the same image data in several resolutions to enable efficient streaming. There is a separate \*.tfrecords file for each resolution, and if the dataset contains labels, they are stored in a separate file as well.


```bash
$ cd stylegan
$ python dataset_tool.py create_from_images datasets/mars ../datasets/cropped_files_aug
```

At the path `datasets/mars` there are going to be several files, each corresponds to a different resolution.
The training is progressive, first it starts with low resolution images and then it increases till the final size.


### Train

#### Setup train.py

Now we are very close to start our training.
To setup the configuration of the training we need to modify the file `stylegan/train.py`.

1. We have the tfrecords as our dataset. We do this by removing this ine
```python
desc += '-ffhq';     dataset = EasyDict(tfrecord_dir='ffhq');                 train.mirror_augment = True
```
and replacing it with this one
```
desc += '-mars';     dataset = EasyDict(tfrecord_dir='mars');                 train.mirror_augment = False
```

2. We need to specify the number of GPUs that we are going to use. In my particular case I only have one GPU.
I commented this line
```python
desc += '-8gpu'; submit_config.num_gpus = 8; sched.minibatch_base = 32; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
```
and uncommented this one
```python
desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
```

3. Finally I changed the number of images that I used in the training set. In [my particular case](#dataaugmentation) they are 20600 images.
So I changed the number 25000 to 20600 in this line.
```python
train.total_kimg = 25000
```
```python
train.total_kimg = 20600
```

The other default parameters worked fine for me.
So now we are ready to go!

#### Train

The training is progressive. This means that it will start with low size images and it will progressively start to increase the size of the images till it reaches the final size.
The first epochs with small probably will be fast, but don't get excited to fast, because as soon as the images increase their size you will notice the difference.
I remind that for 20600 images of 512px it took more than two weeks of training. So, be patient...

To start the training be sure to have active the virtual env that you created for this project and just run
```bash
python train.py
```
and voilà! The train will start.

You will see some prints with the summary of the architecture of the network, and finally a message saying `Training...`. Now we are done.

Every time an epoch has finised you will see a message like this one:
```
tick 1     kimg 140.3    lod 6.00  minibatch 128  time 3m 36s       sec/tick 163.2   sec/kimg 1.16    maintenance 52.8   gpumem 3.8
```

#### Checkout the training progress.

The train script save some snapshots during the training to show you the progress of the training.
You can check them out in `stylegan/results/00001-sgan-mars-1gpu/`.
You will find network snapshots `network-snapshot-****.pkl` from which you can resume the training, and also synthetic samples `fakes****.png`.


### Interpolate the latent space

Here you can find a notebook to interpolate the latent space and play with that image_square_resize

<script src="https://gist.github.com/ivanlen/1c3d426dbe53509b9b99fd883a818bc5.js"></script>

### Generate a gif of the Training

In this notebook you will find how to generate a gif from the synthetic images produced in the different stages of the training.

<script src="https://gist.github.com/ivanlen/e7662d53420f84778c92b77fdc5a4786.js"></script>




#### Some link

- StyleGAN original paper [https://arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948){:target="_ blank"}
- You need CUDA to be able to run tensorflow with GPU support [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads){:target="_ blank"}










.
