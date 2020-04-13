---
title: "Autoencoders safari: simple implementations of three different types of convolutional autoencoders"
layout: post
date: 2020-04-01 12:00
image: /assets/images/markdown.jpg
headerImage: false
# thumbnail: /assets/posts/thumb-low.jpg
tag:
- convolutional autoencoder
- convolutional variational autoencoder
- convolutional conditional variational autoencoder
- generative models
- DNNs
category: blog
author: ivanlen
description: convolutional autoencoders simple implementations
---


Recently I was trying to implement a convolutional conditional variational autoencoder.
There are multiple implementations and resources of different types of autoencoders around, but I couldn't find a resource with multiple implementation of different __convolutional__ autoencoders.
After reading a couple of articles, tutorials and implementations, I decided to write __[three simple notebooks to show how to implement convolutional autoencoders](https://github.com/ivanlen/autoencoders_safari)__, a convolutional autoencoders __safari__.

In these three notebooks you will find:

1. A simple convolutional autoencoder, with some visualizations of the latent space and the different layers.
[https://github.com/ivanlen/autoencoders_safari/blob/master/convolutional_autoencoder.ipynb](https://github.com/ivanlen/autoencoders_safari/blob/master/convolutional_autoencoder.ipynb)

<img src="/assets/images/ae_safari/CAE_latent.png"
      alt="latent space"
      width="400"/>


2. A convolutional variational autoencoder, with a simple implementation of the reparametrization trick, a generator to draw new samples and some visualizations of the latent space.
[https://github.com/ivanlen/autoencoders_safari/blob/master/variational_convolutional_autoencoder.ipynb](https://github.com/ivanlen/autoencoders_safari/blob/master/variational_convolutional_autoencoder.ipynb)

<img src="/assets/images/ae_safari/CVA_latent.png" alt="generated" width="400"/>

- A convolutional conditional variational autoencoders, with also a simple implementation of a generator to draw samples for a desired class.
[https://github.com/ivanlen/autoencoders_safari/blob/master/conditional_convolutional_variatinoal_autoencoder.ipynb](https://github.com/ivanlen/autoencoders_safari/blob/master/conditional_convolutional_variatinoal_autoencoder.ipynb)

<img src="/assets/images/ae_safari/CCVA_generator.png" alt="generated" width="400"/>

Link of the repo:
[https://github.com/ivanlen/autoencoders_safari](https://github.com/ivanlen/autoencoders_safari)
