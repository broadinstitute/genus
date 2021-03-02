#!/usr/bin/env python
# coding: utf-8

from genus.model import *
from genus.util_ml import SpecialDataSet #, ConditionalRandomCrop
from genus.util import *

# Check versions
import torch
import numpy
from platform import python_version
print("python_version() ---> ", python_version())
print("torch.__version__ --> ", torch.__version__)

# make sure to fix the randomness at the very beginning
torch.manual_seed(0)
numpy.random.seed(0)

config = load_json_as_dict("./ML_parameters.json")

# Get the training and test data
img_train, seg_mask_train, count_train = load_obj("./data_train.pt")
img_test, seg_mask_test, count_test = load_obj("./data_test.pt")
img_test_out, seg_mask_test_out, count_test_out = load_obj("./ground_truth")  # using the ground truth filename to pass extrapolation test dataset
BATCH_SIZE = config["simulation"]["batch_size"]


train_loader = SpecialDataSet(x=img_train,
                              x_roi=None,
                              y=seg_mask_train,
                              labels=count_train,
                              data_augmentation=None,
                              store_in_cuda=False,
                              batch_size=BATCH_SIZE,
                              drop_last=True,
                              shuffle=True)

test_loader = SpecialDataSet(x=img_test,
                             x_roi=None,
                             y=seg_mask_test,
                             labels=count_test,
                             data_augmentation=None,
                             store_in_cuda=False,
                             batch_size=BATCH_SIZE,
                             drop_last=True,
                             shuffle=True)

test_out_loader = SpecialDataSet(x=img_test_out,
                                 x_roi=None,
                                 y=seg_mask_test_out,
                                 labels=count_test_out,
                                 data_augmentation=None,
                                 store_in_cuda=False,
                                 batch_size=BATCH_SIZE,
                                 drop_last=True,
                                 shuffle=True)

train_batch_example_fig = train_loader.check_batch()

test_batch_example_fig = test_loader.check_batch()

test_out_batch_example_fig = test_out_loader.check_batch()

# Instantiate model, optimizer and checks
vae = CompositionalVae(config)
optimizer = instantiate_optimizer(model=vae, config_optimizer=config["optimizer"])

# Make reference images
index_tmp = torch.tensor([25, 26, 27, 28, 29], dtype=torch.long)
tmp_imgs, tmp_seg, tmp_count = test_loader.load(index=index_tmp)[:3]
tmp_out_imgs, tmp_out_seg, tmp_out_count = test_out_loader.load(index=index_tmp)[:3]
reference_imgs = torch.cat([tmp_imgs, tmp_out_imgs], dim=0)
reference_count = torch.cat([tmp_count, tmp_out_count], dim=0)

if torch.cuda.is_available():
    reference_imgs = reference_imgs.cuda()
imgs_out = vae.inference_and_generator.unet.show_grid(reference_imgs)

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

    vae.prob_corr_factor = linear_interpolation(epoch,
                                                values=config["shortcut_prob_corr_factor"]["values"],
                                                times=config["shortcut_prob_corr_factor"]["times"])

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

                    output: Output = vae.forward(reference_imgs,
                                                 iom_threshold=config["architecture"]["nms_threshold_test"],
                                                 noisy_sampling=True,
                                                 draw_image=True,
                                                 draw_boxes=True,
                                                 draw_boxes_ideal=True,
                                                 draw_bg=True)

                    reference_n_cells_inferred = output.inference.sample_c_k.sum().item()
                    reference_n_cells_truth = reference_count.sum().item()
                    delta_n_cells = reference_n_cells_inferred - reference_n_cells_truth
                    tmp_dict = {"reference_n_cells_inferred": reference_n_cells_inferred,
                                "reference_delta_n_cells": delta_n_cells}
                    history_dict = append_to_dict(source=tmp_dict,
                                                  destination=history_dict)

                    print("segmentation")
                    segmentation: Segmentation = vae.segment(imgs_in=reference_imgs,
                                                             noisy_sampling=True,
                                                             iom_threshold=config["architecture"]["nms_threshold_test"])

                    print("generation test")
                    generated: Output = vae.generate(imgs_in=reference_imgs,
                                                     draw_boxes=True,
                                                     draw_bg=True)

                    test_loss = test_metrics.loss
                    min_test_loss = min(min_test_loss, test_loss)

                    if (epoch % CHECKPOINT_FREQUENCY == 0) and (epoch >= 20):
                        ckpt = vae.create_ckpt(optimizer=optimizer,
                                               epoch=epoch,
                                               history_dict=history_dict)
                    print("Done epoch")
exp.stop()
