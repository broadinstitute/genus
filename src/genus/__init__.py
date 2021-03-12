# -*- coding: utf-8 -*-
r""" This package implements the a general algorithm for unsupervised instance segmentation.
It relies on pytorch and leidenalg to function. 
The implementation is parallel and scales to large images with hundreds of instances.
"""

__version__ = "0.0.1"
from .namedtuple import BB
from .model import CompositionalVae
from .util import load_obj, save_obj, load_yaml_as_dict, save_dict_as_yaml
from .util_ml import Grid_DPP
from .util_vis import (draw_contours_from_labels,
                       movie_from_resolution_sweep)

__all__ = ["cropper_uncropper", "encoders_decoders", "graph_clustering",
           "namedtuple", "non_max_suppression",
           "unet", "unet_parts", "model", "model_parts",
           "util", "util_logging", "util_ml", "util_vis", "util_data"]

