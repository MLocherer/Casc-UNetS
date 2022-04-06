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
sys.path.append('../pymodules')
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
from source.pymodules import confusion_matrix, get_normalisation_fun, get_loss_fun
from source.pymodules import tabulate_conf_matr, tabulate_train_eval_dict, tabulate_5cv_results
from source.pymodules import filter_prostate_contours
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
    switch_use_stage1_to_predict = True
    print("use stage 1 to predict", switch_use_stage1_to_predict)

    if switch_use_stage1_to_predict:
        # Dataset
        # Dataset file path
        ds_dict = {
            # dataset filepath
            'fp': 'empty',
            # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
            'mic': ['pz', 'cg'],
            # include the following classes in the evaluation I nclude C lasses E valuation
            'ice': ['bg', 'pz', 'cg']
        }

        # number of classes
        stage1_num_classes = 2 # bg + prostate
        stage2_num_classes = len(ds_dict['ice'])
    else:
        # Dataset
        # Dataset file path
        ds_dict = {
            # dataset filepath
            'fp': 'empty',
            # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
            'mic': ['pz', 'cg', 'prostate'],
            # include the following classes in the evaluation I nclude C lasses E valuation
            'ice': ['bg', 'pz', 'cg', 'prostate']
        }

        # number of classes
        stage1_num_classes = 2  # bg + prostate
        stage2_num_classes = len(ds_dict['ice']) - 1
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
    # mopa dict now defines the learning rates for stage 2 UNet2 since UNet1 is pretrained and already completed
    mopa_dict_ds = {
        'clahe':{
            'I2CVB': {
                'lr': 5.0e-4,
                'bs': 6,
            },
            'UDE': {
                'lr': 2.0e-4,
                'bs': 4
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
    # ds_name_keys = ['I2CVB',]
    # local remote
    ds_type = 'local'
    # histogram equalisation clahe or plain
    ds_he = 'clahe'
    # stages
    stage1 = 'unet1'
    # model for stage2 either created by groundtruths or by generated stage 1 output
    # attention gate
    stage2_s1 = ['unet1_gt', 'unet1_gen', 'ag']
    stage2 = os.path.join('unet2', stage2_s1[2])
    print('stage 2 type: ', stage2)


    # filter out contours if multiple prostates are discovered
    switch_filter_contours = True
    print(f"filter out contours: {switch_filter_contours}")
    switch_save_images = True
    print(f"save images: {switch_save_images}")

    # confusion matrix over all 5 folds
    conf_matr_5cv = {}
    imgnames_5cv = {}

    for ds_name_key in ds_name_keys:
        # add keys to dicts
        conf_matr_5cv[ds_name_key] = []
        imgnames_5cv[ds_name_key] = []

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
        # ----------------------------------------------------------------------------------------------------------------------
        # prepare k-fold cross validation
        # ----------------------------------------------------------------------------------------------------------------------
        # update mopa dict with correct lr / bs for dataset
        mopa_dict['lr'] = mopa_dict_ds[ds_he][ds_name_key]['lr']
        mopa_dict['bs'] = mopa_dict_ds[ds_he][ds_name_key]['bs']
        print(mopa_dict)

        image_export_path = os.path.join(ds_path_dict[ds_type]['rp'], stage2, 'imgs', ds_name_key)

        if not os.path.exists(image_export_path):
            os.mkdir(image_export_path)

        # load the 5cv patients list from unet1
        with open(unet1_pickle_fpath, 'rb') as f:
            unet1_train_eval_dict = pickle.load(f)

        # load the 5cv patients list from unet1
        with open(unet2_pickle_fpath, 'rb') as f:
            unet2_train_eval_dict = pickle.load(f)

        for fold in range(unet1_train_eval_dict['folds']):
            # reset confusion matrix
            unet2_train_eval_dict['runs'][fold]['conf_matr_list'] = []
            print('\n')
            print(f"---- start fold #: {fold} ----")
            print(50 * "-")
            print('\n')
            test_patients_li = unet1_train_eval_dict['runs'][fold]['test_patients_li']
            print("Testset: ", test_patients_li)

            # ----------------------------------------------------------------------------------------------------------------------
            # Model path for current fold
            # ----------------------------------------------------------------------------------------------------------------------
            unet2_best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage2, unet2_train_eval_dict['runs'][fold]['best_val_model_fpath'])
            print()
            print(f"loading model for stage 2 from: {unet2_best_val_model_fpath}")
            print()
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

            if switch_use_stage1_to_predict:
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
            stage2_model = UNetSlim(3, stage2_num_classes, bilinear=True)
            stage2_model.load_state_dict(torch.load(unet2_best_val_model_fpath))
            stage2_model = stage2_model.to(mopa_dict['dev'])
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
            stage2_model.eval()
            with torch.no_grad():
                # conf_matr_list contains all the confusion matrices for the val_loader
                conf_matr_list = []
                for b_num, batch in enumerate(test_loader):
                    stage1_inputs, stage2_targets, batch_ids = batch
                    stage1_inputs, stage2_targets = stage1_inputs.to(mopa_dict['dev']), stage2_targets.to(mopa_dict['dev'])
                    # -------------------------------------------------------
                    if switch_use_stage1_to_predict:
                        # stage 1 forward
                        stage1_output = stage1_model(stage1_inputs)
                        # dimensions are N x C x H x W
                        stage1_output = normalization(stage1_output)
                        # determine the winner class
                        stage1_output_one_hot = F.one_hot(torch.argmax(stage1_output, dim=1),
                                                          num_classes=stage1_num_classes).permute(0, 3, 1, 2)
                        if switch_filter_contours:
                            for idx in range(stage1_output_one_hot.shape[0]):
                                arr = filter_prostate_contours(stage1_output_one_hot[idx][1], convex_hull=True)
                                stage1_output_one_hot[idx][1] = torch.tensor(arr).to(mopa_dict['dev'])

                        # create the stage2 inputs
                        stage2_inputs = stage1_inputs * stage1_output_one_hot[:, 1].unsqueeze(dim=1)
                    else:
                        stage2_inputs = stage1_inputs * stage2_targets[:, 3].unsqueeze(dim=1)
                        stage2_targets = stage2_targets[:, :3]

                    # stage 2
                    stage2_output = stage2_model(stage2_inputs)
                    # criterion
                    loss = loss_fn(stage2_output, stage2_targets)
                    unet2_train_eval_dict['runs'][fold]['val_loss_list'].append(loss.cpu().detach().numpy())
                    # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
                    # later calculate the average
                    val_loss += loss.data.item() * stage1_inputs.shape[0]
                    # evaluation procedure begin
                    stage2_output = normalization(stage2_output)
                    # determine the winner class
                    stage2_output_one_hot = F.one_hot(torch.argmax(stage2_output, dim=1), num_classes=stage2_num_classes).permute(0, 3, 1, 2)

                    imgnames_5cv[ds_name_key].append(batch_ids)

                    conf_matr = confusion_matrix(stage2_output_one_hot, stage2_targets, num_classes=stage2_num_classes, batch=True,
                                        target_one_hot=target_one_hot)
                    conf_matr_list.append(conf_matr)
                    # end evaluation procedure

                    # save predictions
                    if switch_save_images:
                        tpb, fpb, tnb, fnb = conf_matr
                        dsc = 2 * tpb / (2 * tpb + fpb + fnb)
                        dsc = dsc.cpu().numpy()
                        for n in range(stage1_inputs.shape[0]):
                            # shape inputs n x 3 x h x w
                            # shape targets n x num_classes x h x w
                            fname = f"{batch_ids['patient'][n]}_{batch_ids['sample_id'][n]}_{batch_ids['biopsy_region'][n]}"
                            save_name = os.path.join(image_export_path, f"{fname}.png")
                            title = f"{batch_ids['patient'][n]} \n {batch_ids['sample_id'][n]} {batch_ids['biopsy_region'][n]} \n {dsc[n]}"
                            plot_prediction_contour(image=stage1_inputs.cpu()[n][1], groundtruth_seg=stage2_targets.cpu()[n],
                                                    prediction_seg=stage2_output_one_hot.cpu()[n],
                                                    include_classes=ds_dict['ice'], show_ticks=False, title=title,
                                                    save_name=save_name, save_only=True)

                    # Progress info
                    sys.stdout.write(
                        f"Validation progress: {(((b_num + 1) * stage1_inputs.shape[0]) / len(testset)) * 100:.2f} % "
                        f"Validation loss (batch): {loss.data.item():.3f} \r")
                    sys.stdout.flush()

                # my evaluation procedure begin
                tp_sum = torch.zeros(stage2_num_classes).to(mopa_dict['dev'])
                fp_sum = torch.zeros(stage2_num_classes).to(mopa_dict['dev'])
                tn_sum = torch.zeros(stage2_num_classes).to(mopa_dict['dev'])
                fn_sum = torch.zeros(stage2_num_classes).to(mopa_dict['dev'])

                # sum up over all batches
                for tp, fp, tn, fn in conf_matr_list:
                    tp_sum += torch.sum(tp, dim=0)
                    fp_sum += torch.sum(fp, dim=0)
                    tn_sum += torch.sum(tn, dim=0)
                    fn_sum += torch.sum(fn, dim=0)
                print("\n")
                print("Confusion matrix according to class: ")
                tabulate_conf_matr((tp_sum, fp_sum, tn_sum, fn_sum), ds_dict['ice'])
                print()

                # calculate average loss
                val_loss /= len(testset)
                print(f"Validation Loss: {val_loss:.2f}  | Duration: {(time.time() - start_time_eval):.2f} sec")

                # plot the prediction
                print("Plot of last prediction (test set) and target:")

            # finish up fold
            stop_time_fold = time.time()
            unet2_train_eval_dict['runs'][fold]['duration_total'] = stop_time_fold - start_time_fold
            print(f"total duration of fold #{fold}: {unet2_train_eval_dict['runs'][fold]['duration_total']:.2f} sec")
            print(
                f"best val loss: {unet2_train_eval_dict['runs'][fold]['best_val_loss']:.2f} in epoch: {unet2_train_eval_dict['runs'][fold]['best_val_epoch']}")
            print()

            # free up memory
            if switch_use_stage1_to_predict:
                del stage1_model
            del stage2_model
            torch.cuda.empty_cache()

            unet2_train_eval_dict['runs'][fold]['conf_matr_list'] = conf_matr_list
            conf_matr_5cv[ds_name_key].extend(conf_matr_list)

        # general evaluation over all folds
        # print('average values for the best epochs over all folds:')
        tabulate_train_eval_dict(train_eval_dict=unet2_train_eval_dict, labels=ds_dict['ice'],
                                 num_classes=stage2_num_classes, device=mopa_dict['dev'])

    # slice wise evaluation
    include_classes = [1, 2] # pz, cg
    tabulate_5cv_results(conf_matr_5cv, imgnames_5cv, include_classes, stage2_num_classes, show_violin_plot=True)