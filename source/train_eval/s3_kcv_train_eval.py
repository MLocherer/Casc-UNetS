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
import math
import time
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# Dataset
from source.pymodules import Compose, ToTensor, RandomHorizontalFlip, Resize, RandomCrop, \
    RandomRotation, RandomGaussianNoise
from source.pymodules import DsI2CVB, calc_i2cvb_weights
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, get_loss_fun
from source.pymodules import tabulate_conf_matr, tabulate_train_eval_dict

from source.pymodules import ds_path_dict

# print only 3 decimals in numpy arrays and suppress scientific notation
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------------------------------------------------------
# Train and evaluate UNET2 on remote server
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------
# begin config

# switch whether to use unet1 to predict the prostate class
# if false the prostate is taken from the groundtruth
switch_use_stage12_to_predict = False

if switch_use_stage12_to_predict:
    # Dataset
    # Dataset file path
    ds_dict = {
        # dataset filepath
        'fp': 'empty',
        # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
        'mic': ['cap', 'prostate'],
        # include the following classes in the evaluation I nclude C lasses E valuation
        'ice': ['bg', 'cap', 'prostate']
    }

    # number of classes
    stage1_num_classes = 2 # bg + prostate
    stage2_num_classes = 3 # bg + pz + cg
    stage3_num_classes = 2 # bg + cap
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
    'clahe':{
        'I2CVB': {
            'lr': 5.0e-4,
            'bs': 8,
        },
        'UDE': {
            'lr': 2.0e-4,
            'bs': 10
        },
        'combined': {
            'lr': 5.0e-4,
            'bs': 8
        }
    }
}

normalization = get_normalisation_fun(mopa_dict['norm'])

# end config
# ----------------------------------------------

# ----------------------------------------------
# begin dataset config
# used to generating pro, pz, cg
train_compose_12 = Compose([
    ToTensor(),
    Resize((320, 320))
])

# used to generate cap
train_compose_3 = Compose([
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
    print("use stage 1 to predict", switch_use_stage12_to_predict)
    switch_augment = False
    print("augment", switch_augment)
    # change the key to select dataset
    ds_name_keys = ['I2CVB', 'UDE', 'combined']
    ds_name_keys = ['I2CVB']
    # local remote
    ds_type = 'remote'
    # histogram equalisation clahe or plain
    ds_he = 'clahe'
    # stages
    stage1 = 'unet1'
    stage2 = 'unet2'
    stage3 = 'unet3'
    for ds_name_key in ds_name_keys:
        print("Load dataset: ", ds_name_key)
        # update file path
        ds_dict['fp'] = ds_path_dict[ds_type][ds_he][ds_name_key]
        print(f"Dataset filepath: {ds_dict['fp']}")
        # unet1 pickle filename
        unet1_run_pickle_fname = f"unet1_{ds_name_key}_{ds_he}_5cv.pickle"
        unet1_pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage1, unet1_run_pickle_fname)
        print()
        print(f"loading model history: {unet1_pickle_fpath}")
        print()
        # unet2 pickle filename
        unet2_run_pickle_fname = f"unet2_{ds_name_key}_{ds_he}_5cv.pickle"
        unet2_pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage2, unet2_run_pickle_fname)
        print()
        print(f"loading model history: {unet2_pickle_fpath}")
        print()
        # unet3 pickle filename
        unet3_run_pickle_fname = f"unet3_{ds_name_key}_{ds_he}_5cv.pickle"
        unet3_pickle_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage3, unet3_run_pickle_fname)
        print()
        print(f"saving model history: {unet3_pickle_fpath}")
        print()
        # ----------------------------------------------------------------------------------------------------------------------
        # prepare k-fold cross validation
        # ----------------------------------------------------------------------------------------------------------------------
        # update mopa dict with correct lr / bs for dataset
        mopa_dict['lr'] = mopa_dict_ds[ds_he][ds_name_key]['lr']
        mopa_dict['bs'] = mopa_dict_ds[ds_he][ds_name_key]['bs']
        print(mopa_dict)

        # load the 5cv patients list from unet1
        with open(unet1_pickle_fpath, 'rb') as f:
            unet1_train_eval_dict = pickle.load(f)

        with open(unet2_pickle_fpath, 'rb') as f:
            unet2_train_eval_dict = pickle.load(f)

        unet3_train_eval_dict = {
            'folds': unet1_train_eval_dict['folds'],
            'runs': {}
        }

        for fold in range(unet1_train_eval_dict['folds']):
            print('\n')
            print(f"---- start fold #: {fold} ----")
            print(50 * "-")
            print('\n')
            train_patients_li = unet1_train_eval_dict['runs'][fold]['train_patients_li']
            test_patients_li = unet1_train_eval_dict['runs'][fold]['test_patients_li']
            print("Trainset: ", train_patients_li)
            print("Testset: ", test_patients_li)

            # ----------------------------------------------------------------------------------------------------------------------
            # Model path for current fold
            # ----------------------------------------------------------------------------------------------------------------------
            unet3_best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage3, f"unet3_{ds_name_key}_{ds_he}_5cv_f{fold}.pt")
            print()
            print(f"saving model to: {unet3_best_val_model_fpath}")
            print()
            # ----------------------------------------------------------------------------------------------------------------------
            # Dataset & Dataloaders
            # ----------------------------------------------------------------------------------------------------------------------
            trainset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=train_patients_li, mr_sequence="T2W",
                               transform=train_compose_12, num_of_surrouding_imgs=1,
                               include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                               samples_must_include_classes=ds_dict['mic'])

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
            if mopa_dict['lf'] in ['GDL', 'WCE']:
                print("Compute class weights")
                weights = calc_i2cvb_weights(I2CVB_dataset=trainset, include_classes=ds_dict['ice'],
                                             target_one_hot=target_one_hot)
                print(f"Compute class weights finished: {weights}")
                weights = weights.to(mopa_dict['dev'])

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
                stage1_model = UNetSlim(3, stage1_num_classes, bilinear=True)
                stage1_model.load_state_dict(torch.load(unet1_best_val_model_fpath))
                stage1_model = stage1_model.to(mopa_dict['dev'])

                # stage 2
                unet2_best_val_model_fpath = os.path.join(ds_path_dict[ds_type]['rp'], stage2,
                                                          unet2_train_eval_dict['runs'][fold]['best_val_model_fpath'])
                stage2_model = UNetSlim(3 + 1, stage2_num_classes, bilinear=True)
                stage2_model.load_state_dict(torch.load(unet2_best_val_model_fpath))
                stage2_model = stage2_model.to(mopa_dict['dev'])

            # stage 3
            stage3_model = UNetSlim(3 + 3, stage3_num_classes, bilinear=True)
            stage3_model = stage3_model.to(mopa_dict['dev'])

            # optim only for stage 2
            # optimizer
            optimizer = optim.Adam(stage3_model.parameters(), lr=mopa_dict['lr'])

            # scheduler for stepsize decay
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            # start of training and evaluation
            # init train_eval_dict for new fold
            unet3_train_eval_dict['runs'][fold] = {
                'train_patients_li': train_patients_li.copy(),
                'test_patients_li': test_patients_li.copy(),
                'best_val_model_fpath': os.path.basename(unet3_best_val_model_fpath),
                'best_train_loss': math.inf,
                'best_train_epoch': -1,
                'train_loss_list': [],
                'best_val_loss': math.inf,
                'best_val_epoch': -1,
                'val_loss_list': [],
                'conf_matr_list': [],
                'duration_total': -1.0
            }

            # use to stop early if epochs_since_last_improvement is higher than early stop patience
            epochs_since_last_improvement = 0
            start_time_fold = time.time()

            # ----------------------------------------------------------------------------------------------------------------------
            # Training and evaluation
            # ----------------------------------------------------------------------------------------------------------------------
            for epoch in range(mopa_dict['epochs']):
                # modify trainset in each epoch create training loader
                train_loader = DataLoader(trainset, batch_size=mopa_dict['bs'], shuffle=True)

                # measure time
                start_time_train = time.time()
                print(f"---- start epoch #: {epoch} ----")

                epochs_since_last_improvement += 1
                train_loss = 0.0
                val_loss = 0.0

                print("Start Training:")

                # Training
                # always keep stage 1 in eval mode
                if switch_use_stage12_to_predict:
                    stage1_model.eval()
                    stage2_model.eval()
                stage3_model.train()
                for i, batch in enumerate(train_loader):
                    # zero out optimizer
                    optimizer.zero_grad()
                    stage1_inputs, stage3_targets, _ = batch
                    stage1_inputs, stage3_targets = stage1_inputs.to(mopa_dict['dev']), stage3_targets.to(mopa_dict['dev'])
                    # MRI, bg, cap, prostate <- prostate so we have all slices with the prostate
                    # print('s1 input', stage1_inputs.shape, 's3 target', stage3_targets.shape)
                    # switch_use_stage12_to_predict = True: s1 input torch.Size([8, 3, 320, 320]) s3 target torch.Size([8, 3, 320, 320])
                    # switch_use_stage12_to_predict = False: s1 input torch.Size([8, 3, 320, 320]) s3 target torch.Size([8, 5, 320, 320])
                    # --------------------------------------------------------
                    if switch_use_stage12_to_predict:
                        # stage 1 forward
                        stage1_output = stage1_model(stage1_inputs)
                        # dimensions are N x C x H x W
                        stage1_output = normalization(stage1_output)
                        # determine the winner class
                        # bg + prostate
                        stage1_output_one_hot = F.one_hot(torch.argmax(stage1_output, dim=1), num_classes=stage1_num_classes).permute(0, 3, 1, 2)
                        # create the stage2 inputs
                        # MRI + prostate
                        stage2_inputs = torch.cat((stage1_inputs, stage1_output_one_hot[:, 1].unsqueeze(dim=1)), dim=1)
                        # print('s2 input', stage2_inputs.shape)
                        # stage 2
                        stage2_output = stage2_model(stage2_inputs)
                        stage2_output = normalization(stage2_output)
                        stage2_output_one_hot = F.one_hot(torch.argmax(stage2_output, dim=1), num_classes=stage2_num_classes).permute(0, 3, 1, 2)
                        # print('s2 output', stage2_output_one_hot.shape)
                        # bg + pz + cg
                        # stage 3 inputs
                        # cap, pro, pz, cg
                        # _a before augmentation
                        stage3_targets_a = torch.cat((stage3_targets[:, 1].unsqueeze(dim=1),                # cap
                                                      stage1_output_one_hot[:, 1].unsqueeze(dim=1),         # pro
                                                      stage2_output_one_hot[:, 1:]),                        # pz + cg
                                                     dim=1)
                        # print('stage3_targets_a', stage3_targets_a.shape)
                        if switch_augment:
                            # send to cpu due to augmentation
                            stage1_inputs, stage3_targets_a = stage1_inputs.cpu(), stage3_targets_a.cpu()
                            for x in range(stage1_inputs.shape[0]):
                                stage1_inputs[x], stage3_targets_a[x] = train_compose_3(stage1_inputs[x], stage3_targets_a[x])
                            stage1_inputs, stage3_targets_a = stage1_inputs.to(mopa_dict['dev']), stage3_targets_a.to(
                                mopa_dict['dev'])

                        stage3_inputs = torch.cat((stage1_inputs, stage3_targets_a[:, 1:]), dim=1)
                        # background must be recreated !!!
                        stage3_targets = torch.cat(((stage3_targets_a[:, 0].unsqueeze(dim=1) == 0) * 1,  # new bg
                                                    stage3_targets_a[:, 0].unsqueeze(dim=1)),  # cap
                                                   dim=1)
                        # testplot
                        """
                        inputs_np_0 = stage3_inputs[0].cpu().numpy()
                        inputs_np_img_1 = np.concatenate((inputs_np_0[0], inputs_np_0[1], inputs_np_0[2]),
                                                       axis=1)
                        inputs_np_img_2 = np.concatenate((inputs_np_0[3], inputs_np_0[4], inputs_np_0[5]),
                                                       axis=1)
                        inputs_np_img = np.concatenate((inputs_np_img_1, inputs_np_img_2), axis=0)
                        plt.imshow(inputs_np_img, cmap="gray")
                        save_fig_path = os.path.join(ds_path_dict[ds_type]['rp'], stage3, f"{i}_test.png")
                        plt.savefig(save_fig_path)
                        """
                        # print('stage3_inputs', stage3_inputs.shape, 'stage3_targets', stage3_targets.shape)
                        # stage3_inputs torch.Size([8, 6, 320, 320]) stage3_targets torch.Size([8, 2, 320, 320])
                    else:
                        stage3_targets_a = stage3_targets[:, 1:]            # all (cap, pz, cg, prostate) except bg

                        if switch_augment:
                            stage1_inputs, stage3_targets_a = stage1_inputs.cpu(), stage3_targets_a.cpu()
                            for x in range(stage1_inputs.shape[0]):
                                stage1_inputs[x], stage3_targets_a[x] = train_compose_3(stage1_inputs[x],
                                                                                        stage3_targets_a[x])
                            stage1_inputs, stage3_targets_a = stage1_inputs.to(mopa_dict['dev']), stage3_targets_a.to(
                                mopa_dict['dev'])

                        stage3_inputs = torch.cat((stage1_inputs,
                                                   stage3_targets_a[:, 3].unsqueeze(dim=1),
                                                   stage3_targets_a[:, 1:3]), dim=1)
                        stage3_targets = torch.cat(
                            ((stage3_targets_a[:, 0].unsqueeze(dim=1) == 0) * 1,  # new bg inverse of cap
                             stage3_targets_a[:, 0].unsqueeze(dim=1)),  # cap
                            dim=1)
                        # testplot
                        """
                        inputs_np_0 = stage3_inputs[0].cpu().numpy()
                        inputs_np_img_1 = np.concatenate((inputs_np_0[0], inputs_np_0[1], inputs_np_0[2]),
                                                       axis=1)
                        inputs_np_img_2 = np.concatenate((inputs_np_0[3], inputs_np_0[4], inputs_np_0[5]),
                                                       axis=1)
                        inputs_np_img = np.concatenate((inputs_np_img_1, inputs_np_img_2), axis=0)
                        plt.imshow(inputs_np_img, cmap="gray")
                        save_fig_path = os.path.join(ds_path_dict[ds_type]['rp'], stage3, f"{i}_test.png")
                        plt.savefig(save_fig_path)
                        """
                        # print('s3 inputs', stage3_inputs.shape, 's3 targets', stage3_targets.shape)
                        # s3 inputs torch.Size([8, 6, 320, 320]) s3 targets torch.Size([8, 2, 320, 320])
                    # --------------------------------------------------------
                    # stage 2
                    stage3_output = stage3_model(stage3_inputs)
                    # for
                    # criterion
                    loss = loss_fn(stage3_output, stage3_targets)
                    unet3_train_eval_dict['runs'][fold]['train_loss_list'].append(loss.cpu().detach().numpy())
                    train_loss += loss.data.item() * stage1_inputs.shape[0]

                    # update learnable parameters
                    loss.backward()
                    optimizer.step()

                    # print out loss in same line `\r` (w/ carriage return sends the cursor to the beginning of the line)
                    sys.stdout.write(
                        f"Progress: {(((i + 1) * stage3_inputs.shape[0]) / len(trainset)) * 100:.2f} % Training loss: {loss.data.item():.3f} \r")
                    sys.stdout.flush()

                # calculate average training loss first
                train_loss /= len(trainset)
                print(f"Avg training Loss: {train_loss:.2f}  | Duration: {(time.time() - start_time_train):.2f} sec")

                # Evaluation
                print()
                print("Start Evaluation:")
                start_time_eval = time.time()
                stage3_model.eval()
                with torch.no_grad():
                    # conf_matr_list contains all the confusion matrices for the val_loader
                    conf_matr_list = []
                    for b_num, batch in enumerate(test_loader):
                        stage1_inputs, stage3_targets, _ = batch
                        stage1_inputs, stage3_targets = stage1_inputs.to(mopa_dict['dev']), stage3_targets.to(mopa_dict['dev'])

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

                            # print('stage3_targets_a', stage3_targets_a.shape)
                            stage3_inputs = torch.cat((stage1_inputs,
                                                       stage1_output_one_hot[:, 1].unsqueeze(dim=1),  # pro
                                                       stage2_output_one_hot[:, 1:]),  # pz + cg
                                                      dim=1)
                            # background must be recreated !!!
                            stage3_targets = stage3_targets[:, :2] # bg + cap without prostate
                            # print('stage3_inputs', stage3_inputs.shape, 'stage3_targets', stage3_targets.shape)
                            # stage3_inputs torch.Size([8, 6, 320, 320]) stage3_targets torch.Size([8, 2, 320, 320])
                        else:
                            stage3_targets_a = torch.cat((stage3_targets[:, 4].unsqueeze(dim=1),  # pro
                                                          stage3_targets[:, 2:4]), dim=1)  # pz, cg
                            stage3_inputs = torch.cat((stage1_inputs, stage3_targets_a), dim=1)
                            stage3_targets = stage3_targets[:, :2]  # bg + cap
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
                        stage3_output_one_hot = F.one_hot(torch.argmax(stage3_output, dim=1), num_classes=stage3_num_classes).permute(0, 3, 1, 2)

                        conf_matr_list.append(confusion_matrix(stage3_output_one_hot,
                                                               stage3_targets,
                                                               num_classes=stage3_num_classes,
                                                               batch=True,
                                                               target_one_hot=target_one_hot))
                        # end evaluation procedure

                        # Progress info
                        sys.stdout.write(
                            f"Validation progress: {(((b_num + 1) * stage3_inputs.shape[0]) / len(testset)) * 100:.2f} % "
                            f"Validation loss (batch): {loss.data.item():.3f} \r")
                        sys.stdout.flush()

                    # my evaluation procedure begin
                    # calculate the total number of tp, fp, tn, fn over the complete dataset. We obtain a vector of tps for
                    # the true positives, fp, ... containing the numbers for each class
                    # calculate the total number of tp, fp, tn, fn over the complete dataset
                    # initialise sums w/ zeros according to the number of classes
                    tp_sum = torch.zeros(stage3_num_classes).to(mopa_dict['dev'])
                    fp_sum = torch.zeros(stage3_num_classes).to(mopa_dict['dev'])
                    tn_sum = torch.zeros(stage3_num_classes).to(mopa_dict['dev'])
                    fn_sum = torch.zeros(stage3_num_classes).to(mopa_dict['dev'])

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

                # update model state_dict files on disk

                # update training parameters
                if unet3_train_eval_dict['runs'][fold]['best_train_loss'] > train_loss:
                    unet3_train_eval_dict['runs'][fold]['best_train_loss'] = train_loss
                    unet3_train_eval_dict['runs'][fold]['best_train_epoch'] = epoch
                # update validation parameters
                if unet3_train_eval_dict['runs'][fold]['best_val_loss'] > val_loss:
                    unet3_train_eval_dict['runs'][fold]['best_val_loss'] = val_loss
                    unet3_train_eval_dict['runs'][fold]['best_val_epoch'] = epoch
                    unet3_train_eval_dict['runs'][fold]['conf_matr_list'] = conf_matr_list
                    epochs_since_last_improvement = 0
                    # save model to drive
                    torch.save(stage3_model.state_dict(), unet3_best_val_model_fpath)

                if mopa_dict['es'] and epochs_since_last_improvement >= mopa_dict['esp']:
                    print(f" Early stopping after {mopa_dict['esp']} epochs.")
                    break

                # Decays the learning rate of each parameter group by gamma every step_size epochs
                # https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.StepLR.html
                if scheduler is not None:
                    scheduler.step()

            # finish up fold
            stop_time_fold = time.time()
            unet3_train_eval_dict['runs'][fold]['duration_total'] = stop_time_fold - start_time_fold
            print(f"total duration of fold #{fold}: {unet3_train_eval_dict['runs'][fold]['duration_total']:.2f} sec")
            print(
                f"best val loss: {unet3_train_eval_dict['runs'][fold]['best_val_loss']:.2f} in epoch: {unet3_train_eval_dict['runs'][fold]['best_val_epoch']}")
            print()

            # free up memory
            if switch_use_stage12_to_predict:
                del stage1_model
                del stage2_model
            del stage3_model
            torch.cuda.empty_cache()

        # general evaluation over all folds
        # print('average values for the best epochs over all folds:')
        tabulate_train_eval_dict(train_eval_dict=unet3_train_eval_dict, labels=ds_dict['ice'],
                                 num_classes=stage3_num_classes, device=mopa_dict['dev'])

        # save training results
        with open(unet3_pickle_fpath, 'wb') as f:
            pickle.dump(unet3_train_eval_dict, f, pickle.HIGHEST_PROTOCOL)
