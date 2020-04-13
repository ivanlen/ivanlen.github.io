---
title: "autoencoders safari: simple implementations of three different types of convolutional autoencoders"
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
#
It easy to find implementations of different types of autoencoders, but in general I couldn't find a placer where dfferent __convolutional__ autoencoders where implement. 
After reading a couple of articles, tutorials and implementations, I decided to write __[three simple notebooks to show how to implement convolutional autoencoders](https://github.com/ivanlen/autoencoders_safari)__, a convolutional autoencoders __safari__.

There are three notebooks using the MNIST dataset:
- A simple convolutional autoencoder, with some visualizations of the latent space and the different layers.
- A convolutional variational autoencoder, with a simple implementation of the reparametrization trick, a generator to draw new samples and some visualizations of the latent space.
- A convolutional conditional variational autoencoders, with also a simple implementation of a generator to draw samples for a desired class.

Link:
https://github.com/ivanlen/autoencoders_safari