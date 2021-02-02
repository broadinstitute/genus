import neptune
import torch.nn
import numpy
from typing import Union
import matplotlib.figure
from typing import Optional, List
from neptunecontrib.api import log_chart
from .namedtuple import ConcordanceIntMask
from .util import save_obj

""" This module logs a variety to metrics, images, artifacts (i.e. generic files) using the Neptune library."""


def log_img_only(name: str,
                 fig: matplotlib.figure.Figure,
                 experiment:  Optional[neptune.experiments.Experiment] = None,
                 verbose: bool = False):
    if verbose:
        print("inside log_img_only -> "+name)
    exp = experiment if experiment else neptune
    exp.log_image(name, fig)
    if verbose:
        print("leaving log_img_only -> "+name)


def log_img_and_chart(name: str,
                      fig: matplotlib.figure.Figure,
                      experiment:  Optional[neptune.experiments.Experiment] = None,
                      verbose: bool = False):
    if verbose:
        print("inside log_img_and_chart -> "+name)
    exp = experiment if experiment else neptune
    log_chart(name, fig, exp)
    exp.log_image(name, fig)
    if verbose:
        print("leaving log_img_and_chart -> "+name)


def log_model_summary(model: torch.nn.Module,
                      experiment: Optional[neptune.experiments.Experiment] = None,
                      verbose: bool = False):
    if verbose:
        print("inside log_model_summary")

    exp = experiment if experiment else neptune

    for x in model.__str__().split('\n'):
        # replace leading spaces with '-' character
        n = len(x) - len(x.lstrip(' '))
        exp.log_text("model summary", '-' * n + x)

    if verbose:
        print("leaving log_model_summary")


def log_object_as_artifact(name: str,
                           obj: object,
                           experiment: Optional[neptune.experiments.Experiment] = None,
                           verbose: bool = False):
    if verbose:
        print("inside log_object_as_artifact")

    path = name+".pt"
    save_obj(obj=obj, path=path)
    exp = experiment if experiment else neptune
    exp.log_artifact(path)

    if verbose:
        print("leaving log_object_as_artifact")


def log_matplotlib_as_png(name: str,
                          fig: matplotlib.figure.Figure,
                          experiment: Optional[neptune.experiments.Experiment] = None,
                          verbose: bool = False):
    if verbose:
        print("log_matplotlib_as_png")

    exp = experiment if experiment else neptune
    fig.savefig(name+".png")  # save to local file
    exp.log_image(name, name+".png")  # log file to neptune

    if verbose:
        print("leaving log_matplotlib_as_png")


def log_many_metrics(metrics: Union[dict, tuple],
                     prefix_for_neptune: str = "",
                     keys_exclude: Optional[List[str]] = None,
                     experiment: Optional[neptune.experiments.Experiment] = None,
                     verbose: bool = False):
    """ Log a dictionary or a tuple of metrics into neptune """

    def log_internal(_exp, _key, _value):
        if isinstance(value, float) or isinstance(value, int):
            _exp.log_metric(prefix_for_neptune + _key, _value)
        elif isinstance(value, numpy.ndarray):
            for i, x in enumerate(value):
                _exp.log_metric(prefix_for_neptune + key + "_" + str(i), x)
        elif isinstance(value, torch.Tensor):
            for i, x in enumerate(value):
                _exp.log_metric(prefix_for_neptune + key + "_" + str(i), x.item())
        else:
            print(_key)
            print(type(_value), _value)
            raise Exception

    if verbose:
        print("inside log_many_metrics")

    exp = experiment if experiment else neptune
    keys_exclude = [""] if keys_exclude is None else keys_exclude

    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if key not in keys_exclude:
                log_internal(exp, prefix_for_neptune + key, value)
    elif isinstance(metrics, tuple):
        for key in metrics._fields:
            value = getattr(metrics, key)
            if key not in keys_exclude:
                log_internal(exp, prefix_for_neptune + key, value)
    else:
        raise Exception("metric type not recognized ->", type(metrics))

    if verbose:
        print("leaving log_dict_metrics")


def log_concordance(concordance: ConcordanceIntMask,
                    prefix_for_neptune: str = "",
                    experiment: Optional[neptune.experiments.Experiment] = None,
                    verbose: bool = False):
    if verbose:
        print("inside log_concordance")

    exp = experiment if experiment else neptune
    tmp_dict = {"iou": concordance.iou,
                "mutual_information": concordance.mutual_information,
                "intersection": concordance.intersection_mask.sum().item(),
                "delta_n": concordance.delta_n,
                "matching_instances": concordance.n_reversible_instances}

    log_many_metrics(metrics=tmp_dict, prefix_for_neptune=prefix_for_neptune, experiment=exp)

    if verbose:
        print("leaving log_concordance")


def log_last_ckpt(name: str,
                  ckpt: dict,
                  experiment: Optional[neptune.experiments.Experiment] = None,
                  verbose: bool = False):

    if verbose:
        print("inside log_last_ckpt")

    exp = experiment if experiment else neptune
    path = name+".pt"
    save_obj(obj=ckpt, path=path)
    print("logging artifact")
    exp.log_artifact(path)

    if verbose:
        print("leaving log_last_ckpt")
