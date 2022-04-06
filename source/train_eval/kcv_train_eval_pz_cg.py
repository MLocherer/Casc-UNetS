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
import math
import time
from operator import itemgetter
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
# Dataset
from source.pymodules import Compose, ToTensor, RandomHorizontalFlip, Resize, RandomCrop, \
    RandomRotation, RandomGaussianNoise
from source.pymodules import DsI2CVB, get_I2CVB_dataset, calc_i2cvb_weights
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, \
    get_loss_fun
from source.pymodules import ds_path_dict, tabulate_conf_matr, tabulate_train_eval_dict


# print only 3 decimals in numpy arrays and suppress scientific notation
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------
# begin config

# Dataset
ds_type = 'remote'
ds_he = 'clahe'
ds_name_key = 'UDE'
# Dataset file path
ds_dict = {
    # dataset filepath
    'fp': ds_path_dict[ds_type][ds_he][ds_name_key],
    # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
    'mic': ['prostate'],
    # include the following classes in the evaluation I nclude C lasses E valuation
    'ice': ['bg', 'prostate']
}

# number of classes
num_classes = len(ds_dict['ice']) - 1
target_one_hot = True

# training parameters
# model parameters dictionary
mopa_dict = {
    # learning rate
    'lr': 5.0e-5,
    # batch size
    'bs': 2,
    # epochs
    'epochs': 40,
    # loss function
    'lf': "DL",
    # normalisation
    'norm': "softmax",
    # device
    'dev': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # early stopping
    'es': True,
    # early stopping after esp epochs
    'esp': 100
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
    # prepare k-fold cross validation
    # ----------------------------------------------------------------------------------------------------------------------
    print(f"Dataset filepath: {ds_dict['fp']}")

    # list with all patients
    ps = get_I2CVB_dataset(ds_dict['fp'])
    num_folds = 5
    kf = KFold(n_splits=num_folds, random_state=None, shuffle=False)
    kf.get_n_splits(ps)

    train_eval_dict = {
        'folds': num_folds,
        'runs': {}
    }
    for fold, (train_index, test_index) in enumerate(kf.split(ps)):
        print('\n')
        print(f"---- start fold #: {fold} ----")
        print(50 * "-")
        print('\n')
        train_patients_li = list(itemgetter(*train_index)(ps))
        test_patients_li = list(itemgetter(*test_index)(ps))
        print("Trainset: ", train_patients_li)
        print("Testset: ", test_patients_li)

        # ----------------------------------------------------------------------------------------------------------------------
        # Dataset & Dataloaders
        # ----------------------------------------------------------------------------------------------------------------------
        trainset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=train_patients_li, mr_sequence="T2W",
                           transform=train_compose, num_of_surrouding_imgs=1,
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
        # net = SegNet(3, len(run_config.include_class_labels))
        # net = UNet(3, len(run_config.include_class_labels), bilinear=False)
        model = UNetSlim(3 + 1, num_classes, bilinear=True)

        # load model from file
        # net.load_state_dict(torch.load(run_autoenc_conf.best_val_model_name))
        model = model.to(mopa_dict['dev'])

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=mopa_dict['lr'])

        # scheduler for stepsize decay
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # start of training and evaluation
        # init train_eval_dict for new fold
        train_eval_dict['runs'][fold] = {
            'best_train_loss': math.inf,
            'best_train_epoch': 0,
            'train_loss_list': [],
            'best_val_loss': math.inf,
            'best_val_epoch': 0,
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
            model.train()
            for i, batch in enumerate(train_loader):
                # zero out optimizer
                optimizer.zero_grad()
                inputs, targets = batch

                # add the prostate segmentation to the input
                inputs_e = torch.cat((inputs, targets[:,3].unsqueeze(dim=1)), dim=1)
                targets = targets[:,:3]
                inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                # forward
                output = model(inputs_e)

                # criterion
                loss = loss_fn(output, targets)
                train_eval_dict['runs'][fold]['train_loss_list'].append(loss.cpu().detach().numpy())
                train_loss += loss.data.item() * inputs.shape[0]

                # update learnable parameters
                loss.backward()
                optimizer.step()

                # print out loss in same line `\r` (w/ carriage return sends the cursor to the beginning of the line)
                sys.stdout.write(
                    f"Progress: {(((i + 1) * inputs.shape[0]) / len(trainset)) * 100:.2f} % Training loss: {loss.data.item():.3f} \r")
                sys.stdout.flush()

            # calculate average training loss first
            train_loss /= len(trainset)
            print(f"Avg training Loss: {train_loss:.2f}  | Duration: {(time.time() - start_time_train):.2f} sec")

            # Evaluation
            print()
            print("Start Evaluation:")
            start_time_eval = time.time()
            model.eval()
            with torch.no_grad():
                # conf_matr_list contains all the confusion matrices for the val_loader
                conf_matr_list = []
                for b_num, batch in enumerate(test_loader):
                    inputs, targets = batch
                    # add the prostate segmentation to the input
                    inputs_e = torch.cat((inputs, targets[:, 3].unsqueeze(dim=1)), dim=1)
                    targets = targets[:, :3]

                    inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                    # forward output contains `probabilities` for each class and pixel -> one hot to find the `winning`
                    # class
                    output = model(inputs_e)

                    # criterion
                    loss = loss_fn(output, targets)
                    train_eval_dict['runs'][fold]['val_loss_list'].append(loss.cpu().detach().numpy())
                    # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
                    # later calculate the average
                    val_loss += loss.data.item() * inputs.shape[0]

                    # my evaluation procedure begin
                    # create one hot encoding for the last prediction of the current batch torch.argmax(pred_last,
                    # dim=0) creates a tensor of N x H X W x C w/ the predicted classes in each pixel we want, however,
                    # a one hot encoding of this tensor to be able to compare it against the target since this target is
                    # one hot encoded F.one_hot returns a tensor of shape (*, num_classes) so we have to permute it
                    # dimensions are N x C x H x W
                    output = normalization(output)
                    # output_one_hot = (output >= 0.5) * 1
                    # determine the winner class
                    output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

                    # append confusion matrix to the conf_matr_list for later evaluation
                    # adds for each batch tp, fp, tn, fn per class to conf_matr_list
                    # if batch_size = 2 the following is added
                    #
                    # tp: tensor([[    81,   1124,    214, 161109], [462,    949,    168, 161878]]),
                    # fp: tensor([[949,  35, 655, 697], [503,  65, 387, 452]]),
                    # tn: tensor([[163762, 162043, 163803,   2648], [163712, 163045, 164294,   2134]]),
                    # fn: tensor([[  72, 1662,  192,  410], [ 187,  805,   15,  400]])
                    conf_matr_list.append(
                        confusion_matrix(output_one_hot, targets, num_classes=num_classes, batch=True,
                                         target_one_hot=target_one_hot))
                    # end my evaluation procedure

                    # Progress info
                    sys.stdout.write(
                        f"Validation progress: {(((b_num + 1) * inputs.shape[0]) / len(testset)) * 100:.2f} % "
                        f"Validation loss (batch): {loss.data.item():.3f} \r")
                    sys.stdout.flush()

                # my evaluation procedure begin
                # calculate the total number of tp, fp, tn, fn over the complete dataset. We obtain a vector of tps for
                # the true positives, fp, ... containing the numbers for each class
                # calculate the total number of tp, fp, tn, fn over the complete dataset
                # initialise sums w/ zeros according to the number of classes
                tp_sum = torch.zeros(num_classes).to(mopa_dict['dev'])
                fp_sum = torch.zeros(num_classes).to(mopa_dict['dev'])
                tn_sum = torch.zeros(num_classes).to(mopa_dict['dev'])
                fn_sum = torch.zeros(num_classes).to(mopa_dict['dev'])

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
            if train_eval_dict['runs'][fold]['best_train_loss'] > train_loss:
                train_eval_dict['runs'][fold]['best_train_loss'] = train_loss
                train_eval_dict['runs'][fold]['best_train_epoch'] = epoch
            # update validation parameters
            if train_eval_dict['runs'][fold]['best_val_loss'] > val_loss:
                train_eval_dict['runs'][fold]['best_val_loss'] = val_loss
                train_eval_dict['runs'][fold]['best_val_epoch'] = epoch
                train_eval_dict['runs'][fold]['conf_matr_list'] = conf_matr_list

            if mopa_dict['es'] and epochs_since_last_improvement >= mopa_dict['esp']:
                print(f" Early stopping after {mopa_dict['esp']} epochs.")
                break

            # Decays the learning rate of each parameter group by gamma every step_size epochs
            # https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.StepLR.html
            if scheduler is not None:
                scheduler.step()

        # finish up fold
        stop_time_fold = time.time()
        train_eval_dict['runs'][fold]['duration_total'] = stop_time_fold - start_time_fold
        print(f"total duration of fold #{fold}: {train_eval_dict['runs'][fold]['duration_total']:.2f} sec")
        print(
            f"best val loss: {train_eval_dict['runs'][fold]['best_val_loss']:.2f} in epoch: {train_eval_dict['runs'][fold]['best_val_epoch']}")
        print()

        # free up memory
        del model
        torch.cuda.empty_cache()

    # general evaluation over all folds
    print('average values for the best epochs over all folds:')
    tabulate_train_eval_dict(train_eval_dict=train_eval_dict, labels=ds_dict['ice'], device=mopa_dict['dev'])
