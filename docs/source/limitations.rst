.. _limitations:

Limitation of GENUS?
====================

GENUS is a software package for unsupervised segmnetation of modular images.

Auch
----

Supervised machine learning models are capable of segmenting images into distinct objects but require
pixel-level annotations which are expensive to generate for most datasets. Despite their successes, supervised models
have two fundamental limitations: i) the quality of the results are tight to the quality of the human-generated
annotations used during training which, in some cases, can be very poor and ii) they do not generalize to
datasets which are `different` from the datasets used during training. Models pretrained on one datasets can sometimes
be fine-tuned to achieve good performance on the datasets of interest. However, this step still requires annotations
which can be impossible to obtain at the required quality and quantity. Moreover pretraining on one dataset and
fine-tuning on a different dataset is `impossible` when, for example, the images in the two
datasets have a different number of channels. This situation is quite common in biological applications in which the
image of interest can consist of many channels but the available annotated datasets usually consist of gray-scale
or RGB images. This leaves open the opportunity for developing `unsupervised` model.

