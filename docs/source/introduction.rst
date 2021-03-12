.. _introduction:

What is GENUS?
==============

GENUS is a software package for unsupervised segmnetation of modular images.

Scope and Purpose
-----------------

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
or RGB images. This leaves open the opportunity for developing `unsupervised` (and weakly-supervised) models.

The main purpose of GENUS is to take modular images (i.e. images comprised of many similar repeated units) together with
few user-defined, interpretable parameters and produce a segmentation of the image. The image is automatically
decomposed into the background and the foreground components. The foreground component is additionally separated into
non-overlapping objects. GENUS is a generalist algorithm capable of segmenting a large class of modular images but
it was developed with biological application in mind, namely to segment cells in fluorescent microscopy images.

Why modular images?
-------------------

Unsupervised segmentation of still images into objects is fundamentally an ill-posed task.
The notion of "object" and "background" can assume arbitrarily complex variations. The definition of objects is
also context dependent. In some situations it is appropriate to considered a car as a single object,
while in others the appropriate choice is to consider the four wheels and the vehicle body separately.
The ill-posed nature of unsupervised segmentation is somehow mitigated when considering "modular images",
i.e. images comprised of many similar repeated units, such as cells in a biological tissue or cars in a parking lot.
By specifying the expected number and typical size of the objects it is possible to guide the model to deliver the
desired segmentation.

Graph based approach
--------------------

The segmentation of some images is genuinely ambiguous and no model can consistently produce the desired results.
In the context of biomedical images, for example, it is often the case that
ML models (even when trained with high-quality annotation) deliver over or under segmented results.
For these challenging situations, it would be highly desiderable to have the ability to tune
the segmentation stringency, as a post-processing step.
To this end we implemented a graph-based strategy.
Our ML algorithm can generate a co-objectness graph, in which each node represents a pixel,
and pixels belonging to the same instance have a non-zero connectivity weight. The same image is segmented multiple
times, pixels that belong to the same instance consistently have strong connections while pixels that are sometimes
assigned to different instances will have weaker connections. The consensus segmentation is obtained by performing a
fast modularity-based community detection on the graph using the Leiden algorithm.
The Leiden algorithm admits a resolution parameter which can be roughly understood as a connectivity threshold below
which a community is divided into further sub-communities, i.e. higher resolution leads to more communities.
In our experiments, we have observed that most images are segmented consistently for a wide range of values of the
resolution parameter. However, for few ambiguous regions the resolution parameter can be used as a simple
and intuitive "knob" to tune the level of segmentation without the need to re-train the model.

In GENUS we have implemented two modes for selecting the resolution parameter:
- In the automated mode, the value of the resolution parameter is automatically determined
- In the interactive mode, the user selects a small region in the image and experiments with different resolution
    parameters in order to determine a resolution that produces the desired segmentation.
    The chosen value is then used to perform community detection on the entire graph.

In unambigous situations, this graph-based approach is not needed and each image can be porcessed just once to create
high quality segmentation masks.

Dynamical Hyperparameters
-------------------------

In complex ML models, the loss function usually consists of multiple terms. The relative importance of these terms is
determined by the value of hyper-parameters which need to be finely tuned in
order to achieve good performance. There is no intuitive way to set these hyper-parameters
and one usually has to resort to time-consuming hyper-parameter sweeps and cross-validation.
In GENUS we follow a different strategy. The users specify the ranges

.....

and  rule to set these parameters and

This procedure is  and, frankly, disappointing.

porcedure is time


the relative
importance of these terms need to be calibrated usually via cross-validation. This leads to expensive


determined by

model achieve good performance only
if these terms are balanced.

whose
In gradient-based ML models, the task is ultimately specified via the loss which is designed so that

function whose minimization





User's choices
--------------


Quick start tutorial can be found hre


Modules
-------

The current version of CellBender contains the following modules. More modules will be added in the future:

* ``remove-background``: This module removes counts due to ambient RNA molecules and random barcode swapping from
  (raw) UMI-based scRNA-seq gene-by-cell count matrices. At the moment, only the count matrices produced by the
  CellRanger `count` pipeline is supported. Support for additional tools and protocols will be added in the future.
  A quick-start tutorial can be found