#!/usr/bin/env python
# coding: utf-8

import neptune
from genus.util_logging import log_object_as_artifact, log_model_summary, log_many_metrics
from genus.model import *
from genus.util_vis import show_batch, plot_reconstruction_and_inference, plot_generation, plot_segmentation
from genus.util_data import DatasetInMemory, ImageFolderWithIndex
from genus.util import load_yaml_as_dict, flatten_dict, load_obj, file2ckpt, linear_interpolation, append_to_dict
import tarfile
from functools import partial
from torchvision import transforms

# Check versions
import torch
import numpy
from platform import python_version
print("python_version() ---> ", python_version())
print("torch.__version__ --> ", torch.__version__)


def idx_to_nobject(class_to_idx: dict, x: int):
    idx_to_class = {v: int(k.split("_")[-1]) for k, v in class_to_idx.items()}
    return idx_to_class[x]


# make sure to fix the randomness at the very beginning
torch.manual_seed(0)
numpy.random.seed(0)

# read config file
config = load_yaml_as_dict("./config.yaml")
neptune.set_project(config["neptune_project"])

exp: neptune.experiments.Experiment = \
    neptune.create_experiment(params=flatten_dict(config),
                              upload_source_files=["./main_room3D.py", "./config.yaml"],
                              upload_stdout=True,
                              upload_stderr=True)

# Get the training and test data
with tarfile.open("data_train.pt", mode='r') as tar:
    tar.extractall(path=".")
root_data_dir = "./room3D"

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = ImageFolderWithIndex(root_data_dir+"/train", transform=transform)
test_dataset = ImageFolderWithIndex(root_data_dir+"/test", transform=transform)

# With this transformation the labels correspond to the number of objects
test_dataset.target_transform = partial(idx_to_nobject, test_dataset.class_to_idx)
train_dataset.target_transform = partial(idx_to_nobject, train_dataset.class_to_idx)

batch_size = config["simulation"]["BATCH_SIZE"]
train_loader = DataloaderWithLoad(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataloaderWithLoad(test_dataset, batch_size=batch_size, shuffle=True)

# Visualize example from the train and test datasets
test_img_example = test_loader.load(n_example=10)[:1]
show_batch(test_img_example, n_col=5, title="example test imgs",
           figsize=(12, 6), experiment=exp, neptune_name="example_test_imgs")

train_img_example = train_loader.load(n_example=10)[:1]
show_batch(train_img_example, n_col=5, title="example train imgs",
           figsize=(12, 6), experiment=exp, neptune_name="example_train_imgs")

# Make reference images
index_tmp = torch.tensor([25, 26, 27, 28, 29, 30, 31, 32, 34, 35], dtype=torch.long)
reference_imgs, reference_count = test_loader.load(index=index_tmp)[:2]
reference_imgs_fig = show_batch(reference_imgs, n_col=5, normalize_range=(0.0, 1.0), title="reference imgs",
                                neptune_name="reference_imgs", experiment=exp)

# Instantiate model, optimizer and checks
vae = CompositionalVae(config)
log_model_summary(vae)
optimizer = instantiate_optimizer(model=vae, config_optimizer=config["optimizer"])

if torch.cuda.is_available():
    reference_imgs = reference_imgs.cuda()
imgs_out = vae.inference_and_generator.unet.show_grid(reference_imgs)
unet_grid_fig = show_batch(imgs_out[:, 0], normalize_range=(0.0, 1.0), neptune_name="unet_grid", experiment=exp)

# Check the constraint dictionary
print("simulation type = "+str(config["simulation"]["type"]))
    
if config["simulation"]["type"] == "scratch":
    
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 999999

elif config["simulation"]["type"] == "resume":

    if torch.cuda.is_available():
        ckpt = file2ckpt(path="ckpt.pt", device=None)
    else:
        ckpt = file2ckpt(path="ckpt.pt", device='cpu')

    load_from_ckpt(ckpt=ckpt,
                   model=vae,
                   optimizer=optimizer,
                   overwrite_member_var=True)

    epoch_restart = ckpt.get('epoch', -1)
    history_dict = ckpt.get('history_dict', {})
    try:
        min_test_loss = min(history_dict.get("test_loss", 999999))
    except:
        min_test_loss = 999999

elif config["simulation"]["type"] == "pretrained":

    if torch.cuda.is_available():
        ckpt = file2ckpt(path="ckpt.pt", device=None)
    else:
        ckpt = file2ckpt(path="ckpt.pt", device='cpu')

    load_from_ckpt(ckpt=ckpt,
                   model=vae,
                   optimizer=None,
                   overwrite_member_var=False)
       
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 999999
    
else:
    raise Exception("simulation type is NOT recognized")
    
# instantiate the scheduler if necessary    
if config["scheduler"]["is_active"]:
    scheduler = instantiate_scheduler(optimizer=optimizer, config_scheduler=config["scheduler"])
else:
    scheduler = None


TEST_FREQUENCY = config["simulation"]["TEST_FREQUENCY"]
CHECKPOINT_FREQUENCY = config["simulation"]["CHECKPOINT_FREQUENCY"]
NUM_EPOCHS = config["simulation"]["MAX_EPOCHS"]
torch.cuda.empty_cache()
for delta_epoch in range(1, NUM_EPOCHS+1):
    epoch = delta_epoch+epoch_restart    

    vae.annealing_factor = linear_interpolation(epoch,
                                                values=(1.0, 0.0),
                                                times=config["loss"]["annealing_times"])
    exp.log_metric("annealing_factor", vae.annealing_factor)

    with torch.autograd.set_detect_anomaly(False):
        with torch.enable_grad():
            vae.train()
            # print("process one epoch train")
            train_metrics = process_one_epoch(model=vae,
                                              dataloader=train_loader,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              iom_threshold=config["architecture"]["nms_threshold_train"],
                                              verbose=(epoch == 0))

            with torch.no_grad():
                print("Train " + train_metrics.pretty_print(epoch))
                history_dict = append_to_dict(source=train_metrics,
                                              destination=history_dict,
                                              prefix_exclude="wrong_examples",
                                              prefix_to_add="train_")
                log_many_metrics(metrics=train_metrics,
                                 prefix_for_neptune="train_",
                                 experiment=exp,
                                 keys_exclude=["wrong_examples"],
                                 verbose=False)

                if (epoch % TEST_FREQUENCY) == 0:

                    vae.eval()
                    test_metrics = process_one_epoch(model=vae,
                                                     dataloader=test_loader,
                                                     optimizer=optimizer,
                                                     scheduler=scheduler,
                                                     iom_threshold=config["architecture"]["nms_threshold_test"],
                                                     verbose=(epoch == 0))
                    print("Test  "+test_metrics.pretty_print(epoch))

                    history_dict = append_to_dict(source=test_metrics,
                                                  destination=history_dict,
                                                  prefix_exclude="wrong_examples",
                                                  prefix_to_add="test_")

                    log_many_metrics(metrics=test_metrics,
                                     prefix_for_neptune="test_",
                                     experiment=exp,
                                     keys_exclude=["wrong_examples"],
                                     verbose=False)

                    if len(test_metrics.wrong_examples) > 0:
                        error_index = torch.tensor(test_metrics.wrong_examples[:5], dtype=torch.long)
                    else:
                        error_index = torch.arange(5, dtype=torch.long)
                    error_test_img = test_loader.load(index=error_index)[0].to(reference_imgs.device)

                    error_output: Output = vae.forward(error_test_img,
                                                       iom_threshold=config["architecture"]["nms_threshold_test"],
                                                       noisy_sampling=True,
                                                       draw_image=True,
                                                       draw_boxes=True,
                                                       draw_boxes_ideal=True,
                                                       draw_bg=True)

                    in_out = torch.cat((error_output.imgs, error_test_img.expand_as(error_output.imgs)), dim=0)
                    _ = show_batch(in_out, n_col=in_out.shape[0]//2, title="error epoch="+str(epoch),
                                   experiment=exp, neptune_name="test_errors")

                    output: Output = vae.forward(reference_imgs,
                                                 iom_threshold=config["architecture"]["nms_threshold_test"],
                                                 noisy_sampling=False,
                                                 draw_image=True,
                                                 draw_boxes=True,
                                                 draw_boxes_ideal=True,
                                                 draw_bg=True)

                    plot_reconstruction_and_inference(output, epoch=epoch, prefix="rec_", experiment=exp)
                    reference_n_obj_inferred = (output.inference.sample_prob_k > 0.5).sum().item()
                    reference_n_obj_truth = reference_count.sum().item()
                    delta_n_cells = reference_n_obj_inferred - reference_n_obj_truth
                    tmp_dict = {"reference_n_cells_inferred": reference_n_obj_inferred,
                                "reference_delta_n_cells": delta_n_cells}
                    log_many_metrics(tmp_dict, prefix_for_neptune="test_", experiment=exp)
                    history_dict = append_to_dict(source=tmp_dict,
                                                  destination=history_dict)

                    print("segmentation")
                    segmentation: Segmentation = vae.segment(imgs_in=reference_imgs,
                                                             noisy_sampling=True,
                                                             iom_threshold=config["architecture"]["nms_threshold_test"])
                    plot_segmentation(segmentation, epoch=epoch, prefix="seg_", experiment=exp)

                    print("generation test")
                    generated: Output = vae.generate(imgs_in=reference_imgs,
                                                     draw_boxes=True,
                                                     draw_bg=True)
                    plot_generation(generated, epoch=epoch, prefix="gen_", experiment=exp)

                    test_loss = test_metrics.loss
                    min_test_loss = min(min_test_loss, test_loss)

                    if (epoch % CHECKPOINT_FREQUENCY == 0) and (epoch > 0):
                        ckpt = vae.create_ckpt(optimizer=optimizer,
                                               epoch=epoch,
                                               history_dict=history_dict)
                        log_object_as_artifact(name="last_ckpt", obj=ckpt, experiment=exp)  # log file into neptune
                    print("Done epoch")
exp.stop()
