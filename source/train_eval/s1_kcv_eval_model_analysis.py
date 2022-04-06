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
from source.pymodules import Compose, ToTensor, RandomHorizontalFlip, Resize, RandomCrop, \
    RandomRotation, RandomGaussianNoise
from source.pymodules import DsI2CVB, plot_prediction_contour
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, get_loss_fun
from source.pymodules import compare_models_5cv
from source.pymodules import filter_prostate_contours, count_contours
from source.pymodules import ds_path_dict


# print only 3 decimals in numpy arrays and suppress scientific notation
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------------------------------------------------------
# evaluate UNET1 locally w/o results from s1_kcv_eval.py compare combined model against individual models
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------
# begin config

# Dataset
# Dataset file path
ds_dict = {
    # dataset filepath
    'fp': 'not initialised',
    # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
    'mic': ['prostate'],
    # include the following classes in the evaluation I nclude C lasses E valuation
    'ice': ['bg', 'prostate']
}

# number of classes
num_classes = len(ds_dict['ice'])
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
mopa_dict_ds = {
    'plain': {
        'I2CVB': {
            'lr': 2.0e-4,
            'bs': 2,
        },
        'UDE': {
            'lr': 2.0e-4,
            'bs': 4
        },
        'combined': {
            'lr': 2.0e-4,
            'bs': 4
        }
    },
    'clahe': {
        'I2CVB': {
            'lr': 5.0e-4,
            'bs': 6,
        },
        'UDE': {
            'lr': 2.0e-4,
            'bs': 4
        },
        'combined': {
            'lr': 5.0e-5,
            'bs': 2
        }
    }
}

normalization = get_normalisation_fun(mopa_dict['norm'])

# end config
# ----------------------------------------------

# ----------------------------------------------
# begin dataset config
train_compose = Compose([
    ToTensor(),
    Resize((320, 320)),
    RandomHorizontalFlip(),
    RandomCrop(10),
    RandomRotation(2),
    RandomGaussianNoise(0.0001)
])

test_compose = Compose([
    ToTensor(),
    Resize((320, 320))
])

# end dataset config
# ----------------------------------------------

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------------------------------------------------------------
    # change the key to select dataset
    ds_name_keys = ['I2CVB', 'UDE']
    # local remote
    ds_type = 'local'
    # histogram equalisation clahe or plain
    ds_he = 'clahe'
    # stage
    stage = 'unet1'

    # filter out contours if multiple prostates are discovered
    switch_filter_contours = True
    print(f"filter out contours: {switch_filter_contours}")
    switch_save_images = False
    print(f"save images: {switch_save_images}")

    # confusion matrix over all 5 folds
    conf_matr_5cv = {}
    conf_matr_5cv_combined = {}
    imgnames_5cv = {}

    for ds_name_key in ds_name_keys:
        # add keys to dicts
        conf_matr_5cv[ds_name_key] = []
        conf_matr_5cv_combined[ds_name_key] = []
        imgnames_5cv[ds_name_key] = []
        # ----------------------------------------------------------------------------------------------------------------------
        # Evaluate Training dict
        # ----------------------------------------------------------------------------------------------------------------------
        # pickle filename
        run_pickle_fname = f"unet1_{ds_name_key}_{ds_he}_5cv.pickle"
        pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage, run_pickle_fname)

        # for combined model
        run_pickle_fname_comb = f"unet1_combined_{ds_he}_5cv.pickle"
        pickle_fpath_combined = os.path.join(ds_path_dict[ds_type]['rp'], stage, run_pickle_fname_comb)

        image_export_path = os.path.join(ds_path_dict[ds_type]['rp'], stage, 'imgs', ds_name_key)

        if not os.path.exists(image_export_path):
            os.mkdir(image_export_path)

        with open(pickle_fpath, 'rb') as f:
            train_eval_dict = pickle.load(f)

        # just for the model
        with open(pickle_fpath_combined, 'rb') as f:
            train_eval_dict_combined = pickle.load(f)

        # update file path
        ds_dict['fp'] = ds_path_dict[ds_type][ds_he][ds_name_key]
        print(f"Dataset filepath: {ds_dict['fp']}")
        print()
        print(f"loading model history from: {pickle_fpath}")
        print()
        # ----------------------------------------------------------------------------------------------------------------------
        # prepare k-fold cross validation
        # ----------------------------------------------------------------------------------------------------------------------
        # update mopa dict with correct lr / bs for dataset
        mopa_dict['lr'] = mopa_dict_ds[ds_he][ds_name_key]['lr']
        mopa_dict['bs'] = mopa_dict_ds[ds_he][ds_name_key]['bs']
        print(mopa_dict)

        for fold in range(train_eval_dict['folds']):
            print('\n')
            print(f"---- start fold #: {fold} ----")
            print(50 * "-")
            print('\n')

            test_patients_li_comb = train_eval_dict_combined['runs'][fold]['test_patients_li']
            if ds_name_key == 'I2CVB':
                # patients start with Patient
                test_patients_li = [i for i in test_patients_li_comb if i.startswith('Patient')]
            elif ds_name_key == 'UDE':
                test_patients_li = [i for i in test_patients_li_comb if not i.startswith('Patient')]
            else:
                print("Unknown dataset in use. terminate")
                break

            print("Testset: ", test_patients_li)

            # ----------------------------------------------------------------------------------------------------------------------
            # Model path for current fold load combined model
            # ----------------------------------------------------------------------------------------------------------------------
            best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage, train_eval_dict['runs'][fold]['best_val_model_fpath'])
            best_val_model_fpath_combined = os.path.join(ds_path_dict[ds_type]['rp'], stage,
                                                train_eval_dict_combined['runs'][fold]['best_val_model_fpath'])
            print()
            print(f"loading model from: {best_val_model_fpath} and \n {best_val_model_fpath_combined}")
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
            # net = SegNet(3, len(run_config.include_class_labels))
            # net = UNet(3, len(run_config.include_class_labels), bilinear=False)
            model = UNetSlim(3, num_classes, bilinear=True)

            # load model from file
            model.load_state_dict(torch.load(best_val_model_fpath))
            model = model.to(mopa_dict['dev'])

            model_combined = UNetSlim(3, num_classes, bilinear=True)

            # load model from file
            model_combined.load_state_dict(torch.load(best_val_model_fpath_combined))
            model_combined = model_combined.to(mopa_dict['dev'])

            # ----------------------------------------------------------------------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------------------------------------------------------------------
            print()
            print("Start Evaluation:")
            start_time_eval = time.time()
            total_prostate_contours = np.zeros(1)
            total_prostate_contours_r = np.zeros(1)
            total_prostate_contours_c = np.zeros(1)
            total_prostate_contours_c_r = np.zeros(1)
            model.eval()
            model_combined.eval()
            with torch.no_grad():
                # conf_matr_list contains all the confusion matrices for the val_loader
                conf_matr_list = []
                conf_matr_list_combined = []
                for b_num, batch in enumerate(test_loader):
                    inputs, targets, batch_ids = batch
                    inputs, targets = inputs.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                    # individual
                    # forward output contains `probabilities` for each class and pixel -> one hot to find the `winning`
                    # class
                    output = model(inputs)

                    # evaluation procedure begin
                    output = normalization(output)

                    # determine the winner class
                    output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

                    # filter out largest contours 1 is prostate channel
                    total_prostate_contours += count_contours(output_one_hot, channels=[1])

                    if switch_filter_contours:
                        for idx in range(output_one_hot.shape[0]):
                            arr = filter_prostate_contours(output_one_hot[idx][1])
                            output_one_hot[idx][1] = torch.tensor(arr).to(mopa_dict['dev'])

                    # removed contours
                    total_prostate_contours_r += count_contours(output_one_hot, channels=[1])

                    imgnames_5cv[ds_name_key].append(batch_ids)

                    conf_matr_list.append(confusion_matrix(output_one_hot, targets, num_classes=num_classes,
                                                           batch=True, target_one_hot=target_one_hot))

                    # combined
                    # class
                    output_combined = model_combined(inputs)

                    # evaluation procedure begin
                    output_combined = normalization(output_combined)

                    # determine the winner class
                    output_combined_one_hot = F.one_hot(torch.argmax(output_combined, dim=1),
                                                        num_classes=num_classes).permute(0, 3, 1, 2)

                    # filter out largest contours 1 is prostate channel
                    total_prostate_contours_c += count_contours(output_combined_one_hot, channels=[1])

                    if switch_filter_contours:
                        for idx in range(output_combined_one_hot.shape[0]):
                            arr = filter_prostate_contours(output_combined_one_hot[idx][1])
                            output_combined_one_hot[idx][1] = torch.tensor(arr).to(mopa_dict['dev'])

                    # removed contours
                    total_prostate_contours_c_r += count_contours(output_combined_one_hot, channels=[1])

                    conf_matr_list_combined.append(
                        confusion_matrix(output_combined_one_hot, targets, num_classes=num_classes,
                                         batch=True, target_one_hot=target_one_hot))

                    # save predictions
                    if switch_save_images:
                        for n in range(inputs.shape[0]):
                            # shape inputs n x 3 x h x w
                            # shape targets n x num_classes x h x w
                            title = f"{batch_ids['patient'][n]}_{batch_ids['sample_id'][n]}_{batch_ids['biopsy_region'][n]}"
                            save_name = os.path.join(image_export_path, f"{title}.png")
                            plot_prediction_contour(image=inputs.cpu()[n][1], groundtruth_seg=targets.cpu()[n],
                                                    prediction_seg=output_one_hot.cpu()[n],
                                                    include_classes=ds_dict['ice'], show_ticks=False, title=title,
                                                    save_name=save_name, save_only=True)

                    # Progress info
                    sys.stdout.write(
                        f"Validation progress: {(((b_num + 1) * inputs.shape[0]) / len(testset)) * 100:.2f} % \r")
                    sys.stdout.flush()

                print(f"Duration: {(time.time() - start_time_eval):.2f} sec")

            # finish up fold
            stop_time_fold = time.time()
            print(f"total duration of fold #{fold}: {train_eval_dict['runs'][fold]['duration_total']:.2f} sec")
            print(f"best val loss: {train_eval_dict['runs'][fold]['best_val_loss']:.2f} in epoch: "
                  f"{train_eval_dict['runs'][fold]['best_val_epoch']}")
            print()

            print(f"total length: {len(test_loader.dataset)} prostates predicted: {total_prostate_contours}, "
                  f"prostate filter: {total_prostate_contours_r}")

            print(f"combined: total length: {len(test_loader.dataset)} prostates predicted: {total_prostate_contours_c}, "
                  f"prostate filter: {total_prostate_contours_c_r}")

            # free up memory
            del model
            del model_combined
            torch.cuda.empty_cache()

            # update confusion matrix over all folds
            conf_matr_5cv[ds_name_key].extend(conf_matr_list)
            conf_matr_5cv_combined[ds_name_key].extend(conf_matr_list_combined)

    # slice wise evaluation
    include_classes = [1] # prostate
    compare_models_5cv(conf_matr_5cv, conf_matr_5cv_combined, imgnames_5cv, include_classes, num_classes, show_violin_plot=True)
