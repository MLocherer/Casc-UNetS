r"""
    Copyright (C) 2022  Mark Locherer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys

sys.path.append('pymodules')

import os
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Dataset
from source.pymodules import Compose, ToTensor, Resize
from source.pymodules import DsI2CVB, plot_prediction_contour

# Model
from source.pymodules import UNetSlim

# Loss
from source.pymodules import get_normalisation_fun, get_loss_fun

# Filter
from source.pymodules import filter_prostate_contours, filter_cap

# paths
from source.pymodules import ds_path_dict

# print only 3 decimals in numpy arrays and suppress scientific notation
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------------------------------------------------------
# Train and evaluate UNET2 on remote server
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------------------------
    # begin config

    # switch whether to use unet1 to predict the prostate class
    # if false the prostate is taken from the groundtruth
    switch_use_stage12_to_predict = True
    print("use stage 1 to predict", switch_use_stage12_to_predict)

    if switch_use_stage12_to_predict:
        # Dataset
        # Dataset file path
        ds_dict = {
            # dataset filepath
            'fp': 'empty',
            # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
            'mic': ['cap', 'pz', 'cg', 'prostate'],
            # include the following classes in the evaluation I nclude C lasses E valuation
            'ice': ['bg', 'cap', 'pz', 'cg', 'prostate']
        }

        # number of classes
        stage1_num_classes = 2  # bg + prostate
        stage2_num_classes = 3  # bg + pz + cg
        stage3_num_classes = 2  # bg + cap
    else:
        # Dataset
        # Dataset file path
        ds_dict = {
            # dataset filepath
            'fp': 'empty',
            # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
            'mic': ['cap', 'pz', 'cg', 'prostate'],
            # include the following classes in the evaluation I nclude C lasses E valuation
            'ice': ['bg', 'cap', 'pz', 'cg', 'prostate']
        }

        # number of classes
        stage1_num_classes = 2  # bg + prostate
        stage2_num_classes = 3
        stage3_num_classes = 2  # bg + cap
    target_one_hot = True

    # training parameters
    # model parameters dictionary for all datasets the same
    mopa_dict = {
        # learning rate # fill with values from mopa_dict_ds
        'lr': -1,
        # batch size
        'bs': -1,
        # epochs
        'epochs': 60,
        # loss function
        'lf': "DL",
        # normalisation
        'norm': "softmax",
        # device
        'dev': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # early stopping
        'es': True,
        # early stopping after esp epochs
        'esp': 40
    }

    # individual configurations after hyper parameter tuning
    # mopa dict now defines the learning rates for stage 3 UNet3 since UNet1/2 is pretrained and already completed
    mopa_dict_ds = {
        'clahe': {
            'I2CVB': {
                'lr': 5.0e-4,
                'bs': 8,
            },
            'UDE': {
                'lr': 2.0e-4,
                'bs': 2
            },
            'combined': {
                'lr': 5.0e-4,
                'bs': 2
            }
        }
    }

    normalization = get_normalisation_fun(mopa_dict['norm'])

    # end config
    # ----------------------------------------------

    # ----------------------------------------------
    # begin dataset config

    test_compose = Compose([
        ToTensor(),
        Resize((320, 320))
    ])

    # end dataset config
    # ----------------------------------------------

    # change the key to select dataset
    ds_name_keys = ['I2CVB', 'UDE', 'combined']
    ds_name_keys = ['I2CVB', ]
    # local remote
    ds_type = 'local'
    # histogram equalisation clahe or plain
    ds_he = 'clahe'
    # stages
    stage1 = os.path.join('unet1', 'unet1_gt_no_aug')
    stage1 = 'unet1'
    # model for stage2 either created by groundtruths or by generated stage 1 output
    stage2_s1 = ['unet1_gt', 'unet1_gt_no_aug', 'unet1_gen', 'unet2_gt_no_aug']
    stage2 = os.path.join('unet2', stage2_s1[0])
    print('stage 2 type: ', stage2)
    # stage 3
    stage3_s1 = ['unet3_gt', 'unet3_gen', 'unet3_gt_no_aug']
    stage3 = os.path.join('unet3', stage3_s1[0])
    print('stage 3 type: ', stage3)

    # filter out contours if multiple prostates are discovered
    switch_filter_stage1 = True
    print(f"filter out contours stage1: {switch_filter_stage1}")
    # results are slightly better without filter
    switch_filter_stage3 = False
    print(f"filter out contours stage3: {switch_filter_stage3}")
    switch_save_images = True
    print(f"save images: {switch_save_images}")

    for ds_name_key in ds_name_keys:
        print("Load dataset: ", ds_name_key)
        # update file path
        ds_dict['fp'] = ds_path_dict[ds_type][ds_he][ds_name_key]
        print(f"Dataset filepath: {ds_dict['fp']}")
        # unet1 pickle filename
        unet1_run_pickle_fname = f"unet1_{ds_name_key}_{ds_he}_5cv.pickle"
        unet1_pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage1, unet1_run_pickle_fname)
        print()
        print(f"loading stage 1 model history: {unet1_pickle_fpath}")
        print()
        # unet2 pickle filename
        unet2_run_pickle_fname = f"unet2_{ds_name_key}_{ds_he}_5cv.pickle"
        unet2_pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage2, unet2_run_pickle_fname)
        print()
        print(f"loading stage 2 model history: {unet2_pickle_fpath}")
        print()
        # unet3 pickle filename
        unet3_run_pickle_fname = f"unet3_{ds_name_key}_{ds_he}_5cv.pickle"
        unet3_pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage3, unet3_run_pickle_fname)
        print()
        print(f"loading stage 3 model history: {unet3_pickle_fpath}")
        print()
        # ----------------------------------------------------------------------------------------------------------------------
        # prepare k-fold cross validation
        # ----------------------------------------------------------------------------------------------------------------------
        # update mopa dict with correct lr / bs for dataset
        mopa_dict['lr'] = mopa_dict_ds[ds_he][ds_name_key]['lr']
        mopa_dict['bs'] = mopa_dict_ds[ds_he][ds_name_key]['bs']
        print(mopa_dict)

        image_export_path = os.path.join(ds_path_dict[ds_type]['rp'], stage3, 'imgs2', ds_name_key)

        if not os.path.exists(image_export_path):
            os.mkdir(image_export_path)

        # load unet1
        with open(unet1_pickle_fpath, 'rb') as f:
            unet1_train_eval_dict = pickle.load(f)

        # load unet2
        with open(unet2_pickle_fpath, 'rb') as f:
            unet2_train_eval_dict = pickle.load(f)

        # load the 5cv patients list from unet3
        with open(unet3_pickle_fpath, 'rb') as f:
            unet3_train_eval_dict = pickle.load(f)

        for fold in range(unet1_train_eval_dict['folds']):
            # reset confusion matrix
            unet3_train_eval_dict['runs'][fold]['conf_matr_list'] = []
            print('\n')
            print(f"---- start fold #: {fold} ----")
            print(50 * "-")
            print('\n')
            test_patients_li = unet1_train_eval_dict['runs'][fold]['test_patients_li']
            print("Testset: ", test_patients_li)

            # ----------------------------------------------------------------------------------------------------------------------
            # Dataset & Dataloaders
            # ----------------------------------------------------------------------------------------------------------------------
            testset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=test_patients_li, mr_sequence="T2W",
                              transform=test_compose, num_of_surrouding_imgs=1,
                              include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                              samples_must_include_classes=ds_dict['mic'])

            test_loader = DataLoader(testset, batch_size=mopa_dict['bs'], shuffle=True)

            # ----------------------------------------------------------------------------------------------------------------------
            # Loss functions
            # ----------------------------------------------------------------------------------------------------------------------
            print("Initialise loss function")

            # exclude background
            dl_exclude_classes = [0]
            # dl_exclude_classes = None

            weights = None
            """
            # if weights are used you have to create trainset
            if mopa_dict['lf'] in ['GDL', 'WCE']:
                print("Compute class weights")
                weights = calc_i2cvb_weights(I2CVB_dataset=trainset, include_classes=ds_dict['ice'],
                                             target_one_hot=target_one_hot)
                print(f"Compute class weights finished: {weights}")
                weights = weights.to(mopa_dict['dev'])            
            """

            loss_fn = get_loss_fun(loss_fun_name=mopa_dict['lf'], normalisation_name=mopa_dict['norm'],
                                   weights=weights, exclude_classes=dl_exclude_classes, target_one_hot=target_one_hot)

            # ----------------------------------------------------------------------------------------------------------------------
            # Load Model
            # ----------------------------------------------------------------------------------------------------------------------
            print(f"Loading model on device {mopa_dict['dev']}")

            if switch_use_stage12_to_predict:
                # stage 1
                unet1_best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage1,
                                                          unet1_train_eval_dict['runs'][fold]['best_val_model_fpath'])
                print()
                print(f"loading model for stage 1 from: {unet1_best_val_model_fpath}")
                print()
                stage1_model = UNetSlim(3, stage1_num_classes, bilinear=True)
                stage1_model.load_state_dict(torch.load(unet1_best_val_model_fpath))
                stage1_model = stage1_model.to(mopa_dict['dev'])
                stage1_model.eval()

                # stage 2
                unet2_best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage2,
                                                          unet2_train_eval_dict['runs'][fold]['best_val_model_fpath'])
                print()
                print(f"loading model for stage 2 from: {unet2_best_val_model_fpath}")
                print()
                stage2_model = UNetSlim(3 + 1, stage2_num_classes, bilinear=True)
                stage2_model.load_state_dict(torch.load(unet2_best_val_model_fpath))
                stage2_model = stage2_model.to(mopa_dict['dev'])
                stage2_model.eval()

            # stage 3
            unet3_best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage3,
                                                      unet3_train_eval_dict['runs'][fold]['best_val_model_fpath'])
            print()
            print(f"loading model for stage 3 from: {unet3_best_val_model_fpath}")
            print()
            stage3_model = UNetSlim(3 + 3, stage3_num_classes, bilinear=True)
            stage3_model.load_state_dict(torch.load(unet3_best_val_model_fpath))
            stage3_model = stage3_model.to(mopa_dict['dev'])

            # use to stop early if epochs_since_last_improvement is higher than early stop patience
            start_time_fold = time.time()

            # ----------------------------------------------------------------------------------------------------------------------
            # Testing
            # ----------------------------------------------------------------------------------------------------------------------

            # measure time
            start_time_train = time.time()

            val_loss = 0.0

            # Evaluation
            print()
            print("Start Evaluation:")
            start_time_eval = time.time()
            stage3_model.eval()
            with torch.no_grad():
                for b_num, batch in enumerate(test_loader):
                    stage1_inputs, stage3_targets, batch_ids = batch
                    stage1_inputs, stage3_targets = stage1_inputs.to(mopa_dict['dev']), stage3_targets.to(
                        mopa_dict['dev'])

                    # -------------------------------------------------------
                    if switch_use_stage12_to_predict:
                        # stage 1 forward
                        stage1_output = stage1_model(stage1_inputs)
                        # dimensions are N x C x H x W
                        stage1_output = normalization(stage1_output)
                        # determine the winner class
                        # bg + prostate
                        stage1_output_one_hot = F.one_hot(torch.argmax(stage1_output, dim=1),
                                                          num_classes=stage1_num_classes).permute(0, 3, 1, 2)
                        if switch_filter_stage1:
                            for idx in range(stage1_output_one_hot.shape[0]):
                                arr = filter_prostate_contours(stage1_output_one_hot[idx][1], convex_hull=True)
                                stage1_output_one_hot[idx][1] = arr.clone().detach().to(mopa_dict['dev'])

                        # prostate = stage1_output_one_hot[:, 1].unsqueeze(dim=1)
                        # create the stage2 inputs
                        # MRI + prostate
                        stage2_inputs = torch.cat((stage1_inputs, stage1_output_one_hot[:, 1].unsqueeze(dim=1)),
                                                  dim=1)
                        # print('s2 input', stage2_inputs.shape)
                        # stage 2
                        stage2_output = stage2_model(stage2_inputs)
                        stage2_output = normalization(stage2_output)
                        stage2_output_one_hot = F.one_hot(torch.argmax(stage2_output, dim=1),
                                                          num_classes=stage2_num_classes).permute(0, 3, 1, 2)
                        # print('s2 output', stage2_output_one_hot.shape)
                        # pz, cg = stage2_output_one_hot[:, 1:]
                        # print('stage3_targets_a', stage3_targets_a.shape)
                        stage3_inputs = torch.cat((stage1_inputs,
                                                   stage1_output_one_hot[:, 1].unsqueeze(dim=1),  # pro
                                                   stage2_output_one_hot[:, 1:]),  # pz + cg
                                                  dim=1)
                        # background must be recreated !!!
                        stage3_targets_complete = stage3_targets.clone()
                        stage3_targets = stage3_targets[:, :2]  # bg + cap without prostate
                        # print('stage3_inputs', stage3_inputs.shape, 'stage3_targets', stage3_targets.shape)
                        # stage3_inputs torch.Size([8, 6, 320, 320]) stage3_targets torch.Size([8, 2, 320, 320])
                        if switch_filter_stage3:
                            pro_pz_cg = torch.cat((stage1_output_one_hot[:, 1].unsqueeze(dim=1),
                                                   stage2_output_one_hot[:, 1:]), dim=1)
                            mask_filter_cap = ((torch.sum(pro_pz_cg, dim=1) >= 1) * 1).to(torch.uint8)

                    else:
                        # use groundtruth for prediction in unets3
                        # this code has not yet been tested!
                        stage3_targets_a = torch.cat((stage3_targets[:, 4].unsqueeze(dim=1),  # pro
                                                      stage3_targets[:, 2:4]), dim=1)  # pz, cg
                        stage3_inputs = torch.cat((stage1_inputs, stage3_targets_a), dim=1)
                        stage3_targets_complete = stage3_targets.clone()
                        stage3_targets = stage3_targets[:, :2]  # bg + cap
                        if switch_filter_stage3:
                            mask_filter_cap = ((torch.sum(stage3_targets[:, 2:], dim=1) >= 1) * 1).to(torch.uint8)
                        # print('s3 inputs', stage3_inputs.shape, 's3 targets', stage3_targets.shape)
                        # s3 inputs torch.Size([8, 6, 320, 320]) s3 targets torch.Size([8, 2, 320, 320])

                    # stage 3
                    stage3_output = stage3_model(stage3_inputs)
                    # criterion
                    loss = loss_fn(stage3_output, stage3_targets)
                    unet3_train_eval_dict['runs'][fold]['val_loss_list'].append(loss.cpu().detach().numpy())
                    # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
                    # later calculate the average
                    val_loss += loss.data.item() * stage3_inputs.shape[0]
                    # evaluation procedure begin
                    stage3_output = normalization(stage3_output)
                    # determine the winner class
                    stage3_output_one_hot = F.one_hot(torch.argmax(stage3_output, dim=1),
                                                      num_classes=stage3_num_classes).permute(0, 3, 1, 2)

                    if switch_filter_stage3:
                        stage3_output_one_hot[:, 1] = filter_cap(stage3_output_one_hot[:, 1], mask_filter_cap,
                                                                 convex_hull=True)
                    stage3_output_one_hot = stage3_output_one_hot.to(mopa_dict['dev'])

                    # network output
                    casc_unet_pred = torch.cat((stage1_output_one_hot[:, 0].unsqueeze(dim=1),  # bg
                                                stage3_output_one_hot[:, 1].unsqueeze(dim=1),  # cap
                                                stage2_output_one_hot[:, 1:],  # pz + cg
                                                stage1_output_one_hot[:, 1].unsqueeze(dim=1)),  # pro
                                               dim=1)

                    # save predictions
                    if switch_save_images:
                        for n in range(stage1_inputs.shape[0]):
                            # shape inputs n x 3 x h x w
                            # shape targets n x num_classes x h x w
                            fname = f"{batch_ids['patient'][n]}_{batch_ids['sample_id'][n]}_{batch_ids['biopsy_region'][n]}"
                            save_name = os.path.join(image_export_path, f"{fname}.png")
                            title = f"{batch_ids['patient'][n]} \n {batch_ids['sample_id'][n]} {batch_ids['biopsy_region'][n]}"
                            plot_prediction_contour(image=stage1_inputs.cpu()[n][1],
                                                    groundtruth_seg=stage3_targets_complete.cpu()[n],
                                                    prediction_seg=casc_unet_pred.cpu()[n],
                                                    include_classes=ds_dict['ice'],
                                                    show_ticks=False, title=title,
                                                    save_name=save_name, save_only=True)

                    # Progress info
                    sys.stdout.write(
                        f"Validation progress: {(((b_num + 1) * stage3_inputs.shape[0]) / len(testset)) * 100:.2f} % "
                        f"Validation loss (batch): {loss.data.item():.3f} \r")
                    sys.stdout.flush()

            # free up memory
            if switch_use_stage12_to_predict:
                del stage1_model
                del stage2_model
            del stage3_model
            torch.cuda.empty_cache()
