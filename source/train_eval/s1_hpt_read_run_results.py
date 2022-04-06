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
from source.pymodules import filter_prostate_contours, count_contours

# Dataset
from source.pymodules import Compose, ToTensor, Resize
from source.pymodules import DsI2CVB, calc_i2cvb_weights, plot_prediction_contour
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, get_loss_fun
from source.pymodules import tabulate_conf_matr, print_loss_list, get_best_val_epochs
# train_utils
from source.pymodules import ds_path_dict



if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------
    # General settings
    # ----------------------------------------------------------------------------------------------------------------------
    pickle_runs_dict = {'plain': {'rp': 'nohe/run/path/',
                                  'I2CVB': 'test-2021-09-17_13-41-02.pickle',
                                  'UDE': 'test-2021-09-17_19-28-42.pickle',
                                  'combined': 'test-2021-09-18_06-40-55.pickle'},
                        'clahe': {'rp': 'clahe/run/path/with/models',
                                  'I2CVB': 'test-2021-09-21_10-47-28.pickle',
                                  'UDE': 'test-2021-09-21_16-10-44.pickle',
                                  'combined': 'test-2021-09-22_03-11-59.pickle'}
                        }

    # change the key to select dataset
    ds_name_key = 'I2CVB'
    # local remote
    ds_type = 'local'
    # histogram equalisation clahe or plain
    ds_he = 'plain'

    run_path = pickle_runs_dict[ds_he]['rp']

    pickle_fname = pickle_runs_dict[ds_he][ds_name_key]

    pickle_fpath = os.path.join(run_path, pickle_fname)
    image_export_path = os.path.join(run_path, 'imgs',  os.path.splitext(pickle_fname)[0])

    if not os.path.exists(image_export_path):
        os.mkdir(image_export_path)

    # filter out contours if multiple prostates are discovered
    filter_contours_switch = True
    print(f"filter out contours: {filter_contours_switch}")

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
    num_classes = len(ds_dict['ice'])
    target_one_hot = True

    testset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=test_patients_li, mr_sequence="T2W",
                      transform=test_compose, num_of_surrouding_imgs=1,
                      include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                      samples_must_include_classes=ds_dict['mic'])

    test_loader = DataLoader(testset, batch_size=best_mopa_dict['bs'], shuffle=True)

    model = UNetSlim(3, num_classes, bilinear=True)
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
    total_prostate_contours = np.zeros(1)
    total_prostate_contours_r = np.zeros(1)
    print()
    print("Start final test evaluation:")
    start_time_eval = time.time()
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        # conf_matr_list contains all the confusion matrices for the val_loader
        conf_matr_list = []
        for b_num, batch in enumerate(test_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(best_mopa_dict['dev']), targets.to(best_mopa_dict['dev'])

            # forward output contains `probabilities` for each class and pixel -> one hot to find the `winning`
            # class
            output = model(inputs)

            # criterion
            loss = loss_fn(output, targets)
            # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
            # later calculate the average

            test_loss += loss.data.item() * inputs.shape[0]

            # evaluation procedure begin
            output = normalization(output)
            # determine the winner class
            output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

            # filter out largest contours 1 is prostate channel
            total_prostate_contours += count_contours(output_one_hot, channels=[1])

            if filter_contours_switch:
                for idx in range(output_one_hot.shape[0]):
                    arr = output_one_hot[idx][1]
                    arr = filter_prostate_contours(output_one_hot[idx][1])
                    output_one_hot[idx][1] = torch.tensor(arr).to(best_mopa_dict['dev'])

            # removed contours
            total_prostate_contours_r += count_contours(output_one_hot, channels=[1])

            conf_matr_list.append(
                confusion_matrix(output_one_hot, targets, num_classes=num_classes, batch=True,
                                 target_one_hot=target_one_hot))

            # save predictions
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
            sys.stdout.write(
                f"Test progress: {(((b_num + 1) * inputs.shape[0]) / len(testset)) * 100:.2f} % "
                f"Test loss (batch): {loss.data.item():.3f} \r")
            sys.stdout.flush()

        tp_sum = torch.zeros(num_classes).to(best_mopa_dict['dev'])
        fp_sum = torch.zeros(num_classes).to(best_mopa_dict['dev'])
        tn_sum = torch.zeros(num_classes).to(best_mopa_dict['dev'])
        fn_sum = torch.zeros(num_classes).to(best_mopa_dict['dev'])

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
        test_loss /= len(testset)
        print(f"test Loss: {test_loss:.2f}  | Duration: {(time.time() - start_time_eval):.2f} sec")

        print(f"total length: {len(test_loader.dataset)} prostates predicted: {total_prostate_contours}, prostate filter: {total_prostate_contours_r}")
