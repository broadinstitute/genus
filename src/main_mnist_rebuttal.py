#!/usr/bin/env python
# coding: utf-8

import neptune
# import genus
from genus.util_logging import log_object_as_artifact, log_model_summary, log_img_only, log_many_metrics
from genus.model import *
from genus.util_vis import show_batch, plot_tiling, plot_label_contours, \
    plot_reconstruction_and_inference, plot_generation, plot_segmentation, plot_img_and_seg, plot_concordance
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

params = load_json_as_dict("./ML_parameters.json")

neptune.set_project(params["neptune_project"])
exp: neptune.experiments.Experiment = \
    neptune.create_experiment(params=flatten_dict(params),
                              upload_source_files=["./main_mnist_rebuttal.py", "./ML_parameters.json"],
                              upload_stdout=True,
                              upload_stderr=True)

# Get the training and test data
img_train, seg_mask_train, count_train = load_obj("./data_train.pt")
img_test, seg_mask_test, count_test = load_obj("./data_test.pt")
img_test_out, seg_mask_test_out, count_test_out = load_obj("./ground_truth")  # using the ground truth filename to pass extrapolation test dataset
BATCH_SIZE = params["simulation"]["batch_size"]


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
log_img_only(name="train_batch_example", fig=train_batch_example_fig, experiment=exp)

test_batch_example_fig = test_loader.check_batch()
log_img_only(name="test_batch_example", fig=test_batch_example_fig, experiment=exp)

test_out_batch_example_fig = test_out_loader.check_batch()
log_img_only(name="test_out_batch_example", fig=test_out_batch_example_fig, experiment=exp)

# Instantiate model, optimizer and checks
vae = CompositionalVae(params)
log_model_summary(vae)
optimizer = instantiate_optimizer(model=vae, dict_params_optimizer=params["optimizer"])

# Make reference images
index_tmp = torch.tensor([25,26,27,28,29], dtype=torch.long)
tmp_imgs, tmp_seg, tmp_count = test_loader.load(index=index_tmp)[:3]
tmp_out_imgs, tmp_out_seg, tmp_out_count = test_out_loader.load(index=index_tmp)[:3]
reference_imgs = torch.cat([tmp_imgs, tmp_out_imgs], dim=0)
reference_count = torch.cat([tmp_count, tmp_out_count], dim=0)

reference_imgs_fig = show_batch(reference_imgs, n_col=5, normalize_range=(0.0, 1.0), neptune_name="reference_imgs", experiment=exp)

if torch.cuda.is_available():
    reference_imgs = reference_imgs.cuda()
imgs_out = vae.inference_and_generator.unet.show_grid(reference_imgs)
unet_grid_fig = show_batch(imgs_out[:, 0], normalize_range=(0.0, 1.0), neptune_name="unet_grid", experiment=exp)

# Check the constraint dictionary
print("simulation type = "+str(params["simulation"]["type"]))
    
if params["simulation"]["type"] == "scratch":
    
    epoch_restart = -1
    history_dict = {}
    min_test_loss = 999999

elif params["simulation"]["type"] == "resume":
    
    ckpt = file2ckpt(path="ckpt.pt", device=None)
    # ckpt = file2ckpt(path="ckpt.pt", device='cpu')

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

elif params["simulation"]["type"] == "pretrained":

    ckpt = file2ckpt(path="ckpt.pt", device=None)
    # ckpt = file2ckpt(path="ckpt.pt", device='cpu')

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
if params["optimizer"]["scheduler_is_active"]:
    scheduler = instantiate_scheduler(optimizer=optimizer, dict_params_scheduler=params["optimizer"])
else:
    scheduler = None


TEST_FREQUENCY = params["simulation"]["TEST_FREQUENCY"]
CHECKPOINT_FREQUENCY = params["simulation"]["CHECKPOINT_FREQUENCY"]
NUM_EPOCHS = params["simulation"]["MAX_EPOCHS"]
torch.cuda.empty_cache()
for delta_epoch in range(1, NUM_EPOCHS+1):
    epoch = delta_epoch+epoch_restart    

    vae.prob_corr_factor = linear_interpolation(epoch,
                                                values=params["shortcut_prob_corr_factor"]["values"],
                                                times=params["shortcut_prob_corr_factor"]["times"])
    exp.log_metric("prob_corr_factor", vae.prob_corr_factor)
        
    with torch.autograd.set_detect_anomaly(False):
        with torch.enable_grad():
            vae.train()
            # print("process one epoch train")
            train_metrics = process_one_epoch(model=vae,
                                              dataloader=train_loader,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              iom_threshold=params["architecture"]["nms_threshold_train"],
                                              verbose=(epoch == 0))

            with torch.no_grad():
                print("Train " + train_metrics.pretty_print(epoch))
                history_dict = append_to_dict(source=train_metrics,
                                              destination=history_dict,
                                              prefix_exclude="wrong_examples",
                                              prefix_to_add="train_")
                if exp is not None:
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
                                                     iom_threshold=params["architecture"]["nms_threshold_test"],
                                                     verbose=(epoch == 0))
                    print("Test  "+test_metrics.pretty_print(epoch))

                    history_dict = append_to_dict(source=test_metrics,
                                                  destination=history_dict,
                                                  prefix_exclude="wrong_examples",
                                                  prefix_to_add="test_")
                    if exp is not None:
                        log_many_metrics(metrics=test_metrics,
                                         prefix_for_neptune="test_",
                                         experiment=exp,
                                         keys_exclude=["wrong_examples"],
                                         verbose=False)

                    test_out_metrics = process_one_epoch(model=vae,
                                                         dataloader=test_out_loader,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler,
                                                         iom_threshold=params["architecture"]["nms_threshold_train"],
                                                         verbose=(epoch == 0))
                    print("Test Out "+test_out_metrics.pretty_print(epoch))
                    history_dict = append_to_dict(source=test_out_metrics,
                                                  destination=history_dict,
                                                  prefix_exclude="wrong_examples",
                                                  prefix_to_add="test_out_")
                    if exp is not None:
                        log_many_metrics(metrics=test_out_metrics,
                                         prefix_for_neptune="test_",
                                         experiment=exp,
                                         keys_exclude=["wrong_examples"],
                                         verbose=False)

                    if len(test_metrics.wrong_examples) > 0:
                        error_index = torch.tensor(test_metrics.wrong_examples[:5], dtype=torch.long)
                    else:
                        error_index = torch.arange(5, dtype=torch.long)
                    error_test_img = test_loader.load(index=error_index)[0].to(reference_imgs.device)

                    if len(test_out_metrics.wrong_examples) > 0:
                        error_index = torch.tensor(test_out_metrics.wrong_examples[:5], dtype=torch.long)
                    else:
                        error_index = torch.arange(5, dtype=torch.long)
                    error_test_out_img = test_out_loader.load(index=error_index)[0].to(reference_imgs.device)
                    error_img = torch.cat((error_test_img, error_test_out_img), dim=0)

                    error_output: Output = vae.forward(error_img,
                                                       iom_threshold=params["architecture"]["nms_threshold_test"],
                                                       noisy_sampling=True,
                                                       draw_image=True,
                                                       draw_boxes=True,
                                                       draw_boxes_ideal=True,
                                                       draw_bg=True)

                    in_out = torch.cat((error_output.imgs, error_img.expand_as(error_output.imgs)), dim=0)
                    _ = show_batch(in_out, n_col=in_out.shape[0]//2, title="error epoch="+str(epoch),
                                   experiment=exp, neptune_name="test_errors")

                    output: Output = vae.forward(reference_imgs,
                                                 iom_threshold=params["architecture"]["nms_threshold_test"],
                                                 noisy_sampling=True,
                                                 draw_image=True,
                                                 draw_boxes=True,
                                                 draw_boxes_ideal=True,
                                                 draw_bg=True)

                    plot_reconstruction_and_inference(output, epoch=epoch, prefix="rec_")
                    reference_n_cells_inferred = output.inference.sample_c_kb.sum().item()
                    reference_n_cells_truth = reference_count.sum().item()
                    delta_n_cells = reference_n_cells_inferred - reference_n_cells_truth
                    tmp_dict = {"reference_n_cells_inferred": reference_n_cells_inferred,
                                "reference_delta_n_cells": delta_n_cells}
                    log_many_metrics(tmp_dict, prefix_for_neptune="test_", experiment=exp)
                    history_dict = append_to_dict(source=tmp_dict,
                                                  destination=history_dict)

                    print("segmentation")
                    segmentation: Segmentation = vae.segment(imgs_in=reference_imgs,
                                                             noisy_sampling=True,
                                                             iom_threshold=params["architecture"]["nms_threshold_test"])
                    plot_segmentation(segmentation, epoch=epoch, prefix="seg_", experiment=exp)

                    # Here I could add a measure of agreement with the ground truth
                    #a = segmentation.integer_mask[0, 0].long()
                    #b = reference_seg.long()
                    #print("CHECK", a.shape, a.dtype, b.shape, b.dtype)
                    #concordance_vs_gt = concordance_integer_masks(a,b)
                    #plot_concordance(concordance=concordance_vs_gt, neptune_name="concordance_vs_gt_")
                    #log_concordance(concordance=concordance_vs_gt, prefix="concordance_vs_gt_")

                    print("generation test")
                    generated: Output = vae.generate(imgs_in=reference_imgs,
                                                     draw_boxes=True,
                                                     draw_bg=True)
                    plot_generation(generated, epoch=epoch, prefix="gen_", experiment=exp)

                    test_loss = test_metrics.loss
                    min_test_loss = min(min_test_loss, test_loss)

                    if (test_loss == min_test_loss) or (epoch % CHECKPOINT_FREQUENCY == 0) and (epoch >= 50):
                        ckpt = vae.create_ckpt(optimizer=optimizer,
                                               epoch=epoch,
                                               history_dict=history_dict)
                        log_object_as_artifact(name="last_ckpt", obj=ckpt)  # log file into neptune
                    print("Done epoch")
exp.stop()
