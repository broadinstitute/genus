import neptune.new as neptune

import torch.nn
import numpy
from typing import Union
from typing import Optional, List
from .namedtuple import ConcordanceIntMask
from .util import save_obj

""" This module logs a variety to metrics, images, artifacts (i.e. generic files) using the Neptune library."""


def log_model_summary(model: torch.nn.Module,
                      experiment: Optional[neptune.run.Run],
                      verbose: bool = False):
    if verbose:
        print("inside log_model_summary")

    if experiment is not None:
        for x in model.__str__().split('\n'):
            # replace leading spaces with '-' character
            n = len(x) - len(x.lstrip(' '))
            token = '-' * n + x
            experiment['model_summary'].log(token)

    if verbose:
        print("leaving log_model_summary")


def log_many_metrics(metrics: Union[dict, tuple],
                     experiment: Optional[neptune.run.Run],
                     prefix_for_neptune: str = "",
                     keys_exclude: Optional[List[str]] = None,
                     verbose: bool = False):
    """ Log a dictionary or a tuple of metrics into neptune """

    def log_internal(_exp, _prefix, _key, _value):
        if isinstance(_value, float) or isinstance(_value, int):
            _exp[_prefix + "/" + _key].log(_value)
        elif isinstance(_value, numpy.ndarray):
            for i, x in enumerate(_value):
                _exp[_prefix + "/" + _key + "_" + str(i)].log(x)
        elif isinstance(_value, torch.Tensor):
            for i, x in enumerate(_value):
                _exp[_prefix + "/" + _key + "_" + str(i)].log(x.item())
        else:
            print(_key, type(_value), _value)
            raise Exception

    if verbose:
        print("inside log_many_metrics")

    keys_exclude = [""] if keys_exclude is None else keys_exclude

    if experiment is not None:
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key not in keys_exclude:
                    log_internal(experiment, prefix_for_neptune, key, value)
        elif isinstance(metrics, tuple):
            for key in metrics._fields:
                value = getattr(metrics, key)
                if key not in keys_exclude:
                    log_internal(experiment, prefix_for_neptune, key, value)
        else:
            raise Exception("metric type not recognized ->", type(metrics))

    if verbose:
        print("leaving log_dict_metrics")


def log_concordance(concordance: ConcordanceIntMask,
                    experiment: Optional[neptune.run.Run],
                    prefix_for_neptune: str = "",
                    verbose: bool = False):
    if verbose:
        print("inside log_concordance")

    tmp_dict = {"iou": concordance.iou,
                "mutual_information": concordance.mutual_information,
                "intersection": concordance.intersection_mask.sum().item(),
                "delta_n": concordance.delta_n,
                "matching_instances": concordance.n_reversible_instances}
    if experiment is not None:
        log_many_metrics(metrics=tmp_dict, prefix_for_neptune=prefix_for_neptune, experiment=experiment)

    if verbose:
        print("leaving log_concordance")


def log_object_as_artifact(name: str,
                           obj: object,
                           experiment: Optional[neptune.run.Run],
                           verbose: bool = False):
    if verbose:
        print("inside log_object_as_artifact")

    if experiment is not None:
        path = name+".pt"
        save_obj(obj=obj, path=path)
        experiment[name].upload(path)

    if verbose:
        print("leaving log_object_as_artifact")


# def log_img_only(name: str,
#                  fig: matplotlib.figure.Figure,
#                  experiment:  Optional[neptune.experiments.Experiment],
#                  verbose: bool = False):
#     if verbose:
#         print("inside log_img_only -> "+name)
#
#     if experiment is not None:
#         experiment[name].log(fig)
#
#     if verbose:
#         print("leaving log_img_only -> "+name)
#
# def log_matplotlib_as_png(name: str,
#                           fig: matplotlib.figure.Figure,
#                           experiment: Optional[neptune.experiments.Experiment],
#                           verbose: bool = False):
#     if verbose:
#         print("log_matplotlib_as_png")
#
#     if experiment is not None:
#         fig.savefig(name+".png")  # save to local file
#         experiment.log_image(name, name+".png")  # log file to neptune
#
#     if verbose:
#         print("leaving log_matplotlib_as_png")
