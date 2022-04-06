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
import os
import sys
sys.path.append('../pymodules')
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Dataset
from source.pymodules import Compose, ToTensor, Resize
from source.pymodules import DsI2CVB, calc_i2cvb_weights, plot_prediction_contour, plot_segmentation_contour
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, get_loss_fun
from source.pymodules import print_loss_list, get_best_val_epochs, tabulate_5cv_results
# train_utils
from source.pymodules import ds_path_dict


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------
    # General settings
    # ----------------------------------------------------------------------------------------------------------------------
    pickle_runs_dict = {
        'clahe': {
            'rp': '/path/to/run/path/stage3',
            'I2CVB': 'test-2021-10-18_06-15-01.pickle',
            'UDE': 'test-2021-10-18_11-53-50.pickle',
            'combined': 'test-2021-10-18_23-57-56.pickle'
        }
    }

    # change the key to select dataset
    ds_name_keys = ['I2CVB', 'UDE', 'combined']
    ds_name_keys = ['I2CVB']
    # local remote
    ds_type = 'local'
    # histogram equalisation clahe or plain
    ds_he = 'clahe'

    run_path = pickle_runs_dict[ds_he]['rp']

    switch_save_images = False
    print(f"save images: {switch_save_images}")

    # confusion matrix over all 5 folds
    conf_matr_5cv = {}
    imgnames_5cv = {}

    for ds_name_key in ds_name_keys:
        # add keys to dicts
        conf_matr_5cv[ds_name_key] = []
        imgnames_5cv[ds_name_key] = []

        pickle_fname = pickle_runs_dict[ds_he][ds_name_key]

        pickle_fpath = os.path.join(run_path, pickle_fname)
        image_export_path = os.path.join(run_path, 'imgs',  os.path.splitext(pickle_fname)[0])

        if not os.path.exists(image_export_path):
            os.mkdir(image_export_path)

        # ----------------------------------------------------------------------------------------------------------------------
        # Evaluate Training dict
        # ----------------------------------------------------------------------------------------------------------------------
        with open(pickle_fpath, 'rb') as f:
            train_eval_dict = pickle.load(f)

        # ----------------------------------------------------------------------------------------------------------------------
        # Print loss
        # ----------------------------------------------------------------------------------------------------------------------
        print_loss_list(train_eval_dict, plot_title=ds_name_key)
        best_val_epochs, best_val_loss = get_best_val_epochs(train_eval_dict)

        # ----------------------------------------------------------------------------------------------------------------------
        # Generate images and predictions
        # ----------------------------------------------------------------------------------------------------------------------
        best_val_model_fpath = os.path.join(run_path, os.path.basename(train_eval_dict['best_val_model_fpath']))

        test_compose = Compose([
            ToTensor(),
            Resize((320, 320))
        ])

        # Dataset
        # Dataset file path
        ds_dict = train_eval_dict['ds_dict']

        print()
        print(f"change to local filepath accordingly: dataset = {os.path.basename(ds_dict['fp'])}")
        # update path here
        ds_dict['fp'] = ds_path_dict[ds_type][ds_he][ds_name_key]
        print(f"selected dataset = {os.path.basename(ds_dict['fp'])}")

        global_best_val_test_num = train_eval_dict['global_best_val_test_num']
        best_mopa_dict = train_eval_dict['runs'][global_best_val_test_num]['mopa_dict']
        test_patients_li = train_eval_dict['test_patients_li']

        # number of classes
        num_classes = len(ds_dict['ice']) - 3
        #num_classes = 2
        target_one_hot = True

        testset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=test_patients_li, mr_sequence="T2W",
                          transform=test_compose, num_of_surrouding_imgs=1,
                          include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                          samples_must_include_classes=ds_dict['mic'])

        test_loader = DataLoader(testset, batch_size=best_mopa_dict['bs'], shuffle=True)

        # Unet2 stage (4 input slices 2 outputs)
        model = UNetSlim(3 + 3, num_classes, bilinear=True)

        model.load_state_dict(torch.load(best_val_model_fpath))
        # load model from file
        # net.load_state_dict(torch.load(run_autoenc_conf.best_val_model_name))
        model = model.to(best_mopa_dict['dev'])
        # exclude background
        dl_exclude_classes = [0]
        # dl_exclude_classes = None

        weights = None
        if best_mopa_dict['lf'] in ['GDL', 'WCE']:
            print("Compute class weights")
            weights = calc_i2cvb_weights(I2CVB_dataset=testset, include_classes=ds_dict['ice'],
                                         target_one_hot=target_one_hot)
            print(f"Compute class weights finished: {weights}")
            weights = weights.to(best_mopa_dict['dev'])

        loss_fn = get_loss_fun(loss_fun_name=best_mopa_dict['lf'], normalisation_name=best_mopa_dict['norm'],
                               weights=weights, exclude_classes=dl_exclude_classes, target_one_hot=target_one_hot)

        normalization = get_normalisation_fun(best_mopa_dict['norm'])

        # start final test evaluation
        print()
        print("Start final test evaluation:")
        start_time_eval = time.time()
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for b_num, batch in enumerate(test_loader):
                inputs, targets, batch_ids = batch
                # add the prostate segmentation to the input
                inputs_e = torch.cat((inputs, targets[:, 2:]), dim=1)
                targets = targets[:, :2]
                inputs_e, targets = inputs_e.to(best_mopa_dict['dev']), targets.to(best_mopa_dict['dev'])

                # forward output contains `probabilities` for each class and pixel -> one hot to find the `winning`
                # class
                output = model(inputs_e)
                # criterion
                loss = loss_fn(output, targets)
                # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
                # later calculate the average
                test_loss += loss.data.item() * inputs_e.shape[0]

                # dimensions are N x C x H x W
                output = normalization(output)
                # determine the winner class
                output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

                # add names to lists here it's not 5cv but it works if there is only one fold
                imgnames_5cv[ds_name_key].append(batch_ids)
                conf_matr_5cv[ds_name_key].append(
                    confusion_matrix(output_one_hot, targets, num_classes=num_classes, batch=True,
                                     target_one_hot=target_one_hot))

                # save predictions
                if switch_save_images:
                    for n in range(inputs.shape[0]):
                        # shape inputs n x 3 x h x w
                        # shape targets n x num_classes x h x w
                        title = f"{b_num}_{n}"
                        save_name = os.path.join(image_export_path, f"{title}.png")
                        plot_prediction_contour(image=inputs.cpu()[n][1], groundtruth_seg=targets.cpu()[n],
                                                prediction_seg=output_one_hot.cpu()[n],
                                                include_classes=ds_dict['ice'], show_ticks=False, title=title,
                                                save_name=save_name, save_only=True)

                # Progress info
                sys.stdout.write(f"Test progress: {(((b_num + 1) * inputs_e.shape[0]) / len(testset)) * 100:.2f} % Test loss (batch): {loss.data.item():.3f} \r")
                sys.stdout.flush()
            # calculate average loss
            test_loss /= len(testset)
            print(f"test Loss: {test_loss:.2f}  | Duration: {(time.time() - start_time_eval):.2f} sec")

    include_classes = [1] # cap
    tabulate_5cv_results(conf_matr_5cv, imgnames_5cv, include_classes, num_classes, show_violin_plot=True)