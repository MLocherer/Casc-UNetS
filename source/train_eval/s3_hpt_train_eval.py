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
import math
import time
import datetime
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
# Dataset
from source.pymodules import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, Resize, RandomCrop, \
    RandomRotation, RandomGaussianNoise
from source.pymodules import DsI2CVB, get_I2CVB_dataset, calc_i2cvb_weights
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, get_loss_fun
from source.pymodules import tabulate_conf_matr, RunBuilder
# RunBuilder
from collections import OrderedDict
from source import pymodules as dp

# print only 3 decimals in numpy arrays and suppress scientific notation
np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------
# begin config

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
num_classes = len(ds_dict['ice']) - 3
target_one_hot = True

# training parameters
# model parameters dictionary
mopa_dict = {
    # learning rate
    'lr': 5.0e-5,
    # batch size
    'bs': 2,
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

    datasets = [dp.ds_i2cvb_c_r, dp.ds_ude_c_r, dp.ds_comb_c_r]
    dataset_splits = {dp.ds_i2cvb_c_r: {'train_patients_li': ['Patient 513', 'Patient 384', 'Patient 836', 'Patient 383', 'Patient 804', 'Patient 784', 'Patient 1041', 'Patient 870', 'Patient 430'],
                                        'val_patients_li': ['Patient 782', 'Patient 996', 'Patient 634', 'Patient 410'],
                                        'test_patients_li': ['Patient 387', 'Patient 531', 'Patient 799', 'Patient 1036', 'Patient 416', 'Patient 778']},
                      dp.ds_ude_c_r: {'train_patients_li': ['2.25.20160260637862977496883371611382748412', '2.25.218216477731213362336537663063129048430', '2.25.16587954577960538519952296357807597863', '2.25.47654875590220848581826852294722641736', '2.25.288959812455210728478256593086025361114', '2.25.169499415547834839663877919141036099853', '2.25.62960676618543916833271465846852233054', '2.25.171884663526867358437998356086042322206', '2.25.324379588584499243665925913838906917204', '2.25.154613599351710398354815863125892978281', '2.25.115118562367465760885868556047363006809', '2.25.166568643142668535991132519590311137058', '2.25.36571998564400483058529933346370129061', '2.25.207535082355298937769288472440085100416'],
                                        'val_patients_li': ['2.25.145325119148112227535704660150432112744', '2.25.312232340879547966351204976918670355606', '2.25.98424014757228127147521087817339781720', '2.25.46633705456322350328401117110968071646', '2.25.165299523289576307633218666788917632442', '2.25.97451150620477817939599116248152956482'],
                                        'test_patients_li': ['2.25.163734887344991709486017639730798819954', '2.25.323563794825064776404353844122547564027', '2.25.235354401574461393373225623473746014277', '2.25.171357647892041850694769228383989690160', '2.25.52696973620974230257621093358972518246', '2.25.265499916785337359131195050167479870693', '2.25.266584107538929881008812954206231620606', '2.25.67689433239126820915884526133817064950', '2.25.301126510654538781659991078822966621264']},
                      dp.ds_comb_c_r: {'train_patients_li': ['2.25.171357647892041850694769228383989690160', 'Patient 778', '2.25.218216477731213362336537663063129048430', 'Patient 384', '2.25.16587954577960538519952296357807597863', '2.25.323563794825064776404353844122547564027', '2.25.98424014757228127147521087817339781720', '2.25.235354401574461393373225623473746014277', 'Patient 836', 'Patient 996', '2.25.169499415547834839663877919141036099853', '2.25.171884663526867358437998356086042322206', '2.25.145325119148112227535704660150432112744', '2.25.324379588584499243665925913838906917204', '2.25.154613599351710398354815863125892978281', 'Patient 784', 'Patient 1036', '2.25.265499916785337359131195050167479870693', '2.25.163734887344991709486017639730798819954', 'Patient 870', '2.25.36571998564400483058529933346370129061', '2.25.207535082355298937769288472440085100416', 'Patient 430'],
                                       'val_patients_li': ['Patient 416', '2.25.47654875590220848581826852294722641736', '2.25.166568643142668535991132519590311137058', 'Patient 531', 'Patient 782', '2.25.312232340879547966351204976918670355606', '2.25.301126510654538781659991078822966621264', 'Patient 410', '2.25.165299523289576307633218666788917632442', '2.25.62960676618543916833271465846852233054'],
                                       'test_patients_li':['2.25.46633705456322350328401117110968071646', '2.25.266584107538929881008812954206231620606', '2.25.67689433239126820915884526133817064950', 'Patient 1041', 'Patient 634', '2.25.288959812455210728478256593086025361114', 'Patient 513', '2.25.20160260637862977496883371611382748412', '2.25.97451150620477817939599116248152956482', 'Patient 804', '2.25.115118562367465760885868556047363006809', '2.25.52696973620974230257621093358972518246', 'Patient 799', 'Patient 387', 'Patient 383']}}

    run_path = '/run/basedir/remote'
    # if False select a prexisiting split
    random_split = False
    for ds in datasets:
        ds_dict['fp'] = ds
        print(80 * '-')
        print(f"start with dataset: {os.path.basename(ds)}")
        print(80 * '-')
        # ----------------------------------------------------------------------------------------------------------------------
        # General settings
        # ----------------------------------------------------------------------------------------------------------------------

        model_fname = 'test-{date:%Y-%m-%d_%H-%M-%S}.pickle'.format(date=datetime.datetime.now())
        best_val_model_fpath = os.path.join(run_path, 'model-{date:%Y-%m-%d_%H-%M-%S}.pt'.format(date=datetime.datetime.now()))

        # ----------------------------------------------------------------------------------------------------------------------
        # prepare train / val / test split
        # ----------------------------------------------------------------------------------------------------------------------
        # initialise random seed
        random.seed(30)

        print(f"Dataset filepath: {ds_dict['fp']}")

        if random_split:
            ps = get_I2CVB_dataset(ds_dict['fp'])

            # train, val, test split configuration
            ds_len = len(ps)
            split = {'train': .5, 'val': .2, 'test': .3}

            val_abs = int(math.ceil(split['val'] * ds_len))
            test_abs = int(math.ceil(split['test'] * ds_len))
            train_abs = ds_len - val_abs - test_abs

            val_patients_li = random.sample(ps, val_abs)
            ps = [x for x in ps if x not in val_patients_li]

            test_patients_li = random.sample(ps, test_abs)
            ps = [x for x in ps if x not in test_patients_li]

            train_patients_li = ps.copy()
            del ps
        else:
            train_patients_li = dataset_splits[ds]['train_patients_li']
            val_patients_li = dataset_splits[ds]['val_patients_li']
            test_patients_li = dataset_splits[ds]['test_patients_li']

        print("Trainset: ", train_patients_li)
        print("Valset: ", val_patients_li)
        print("Testset: ", test_patients_li)

        # ----------------------------------------------------------------------------------------------------------------------
        # Dataset & Dataloaders
        # ----------------------------------------------------------------------------------------------------------------------
        trainset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=train_patients_li, mr_sequence="T2W",
                           transform=train_compose, num_of_surrouding_imgs=1,
                           include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                           samples_must_include_classes=ds_dict['mic'])

        valset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=val_patients_li, mr_sequence="T2W",
                         transform=test_compose, num_of_surrouding_imgs=1,
                         include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                         samples_must_include_classes=ds_dict['mic'])

        val_loader = DataLoader(valset, batch_size=mopa_dict['bs'], shuffle=True)

        train_eval_dict = {
            'folds': 'hyperparameter tuning',
            'best_val_model_fpath': best_val_model_fpath,
            'ds_dict': ds_dict.copy(),
            'train_patients_li': train_patients_li,
            'val_patients_li': val_patients_li,
            'test_patients_li': test_patients_li,
            'global_best_val_loss': math.inf,
            'global_best_val_epoch': -1,
            'global_best_val_test_num': -1,
            'final_test_dict': {},
            'runs': {}
        }

        params = OrderedDict(
            lr=[5e-4, 2.0e-4, 1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5],
            bs=[2, 4, 6, 8, 10]
        )

        runs = RunBuilder.get_runs(params)

        for test_num, run in enumerate(runs):
            print('\n')
            print(f"---- start hyperparameter set #: {test_num} ----")
            print(50 * "-")
            print('\n')

            # ----------------------------------------------------------------------------------------------------------------------
            # update mopa dict
            # ----------------------------------------------------------------------------------------------------------------------
            mopa_dict['lr'] = run.lr
            mopa_dict['bs'] = run.bs

            print(mopa_dict)

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

            # init model the model is trained on the three image input + the additional prostate target segmentation
            model = UNetSlim(3 + 3, num_classes, bilinear=True)

            # send to device
            model = model.to(mopa_dict['dev'])

            # optimizer
            optimizer = optim.Adam(model.parameters(), lr=mopa_dict['lr'])

            # scheduler for stepsize decay
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            # start of training and evaluation
            # init train_eval_dict for new fold
            train_eval_dict['runs'][test_num] = {
                'mopa_dict': mopa_dict.copy(),
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
                model.train()
                for i, batch in enumerate(train_loader):
                    # zero out optimizer
                    optimizer.zero_grad()
                    inputs, targets, _ = batch

                    # add the prostate segmentation to the input
                    inputs_e = torch.cat((inputs, targets[:, 2].unsqueeze(dim=1), targets[:, 3].unsqueeze(dim=1), targets[:, 4].unsqueeze(dim=1)), dim=1)
                    targets = targets[:, :2]
                    inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                    # forward
                    output = model(inputs_e)

                    # criterion applies also normalisation
                    loss = loss_fn(output, targets)

                    train_eval_dict['runs'][test_num]['train_loss_list'].append(loss.cpu().detach().numpy())
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
                    for b_num, batch in enumerate(val_loader):
                        inputs, targets, _ = batch
                        inputs, targets = inputs.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                        # add the prostate segmentation to the input
                        inputs_e = torch.cat((inputs, targets[:, 2].unsqueeze(dim=1), targets[:, 3].unsqueeze(dim=1),
                                              targets[:, 4].unsqueeze(dim=1)), dim=1)
                        targets = targets[:, :2]

                        inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                        inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                        output = model(inputs_e)

                        # criterion
                        loss = loss_fn(output, targets)

                        train_eval_dict['runs'][test_num]['val_loss_list'].append(loss.cpu().detach().numpy())
                        # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
                        # later calculate the average
                        val_loss += loss.data.item() * inputs.shape[0]

                        # my evaluation procedure begin
                        # dimensions are N x C x H x W
                        output = normalization(output)
                        # determine the winner class
                        output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

                        # append confusion matrix to the conf_matr_list for later evaluation
                        conf_matr_list.append(
                            confusion_matrix(output_one_hot, targets, num_classes=num_classes, batch=True,
                                             target_one_hot=target_one_hot))
                        # end my evaluation procedure

                        # Progress info
                        sys.stdout.write(
                            f"Validation progress: {(((b_num + 1) * inputs.shape[0]) / len(valset)) * 100:.2f} % "
                            f"Validation loss (batch): {loss.data.item():.3f} \r")
                        sys.stdout.flush()

                    # my evaluation procedure begin
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
                    val_loss /= len(valset)
                    print(f"Validation Loss: {val_loss:.2f}  | Duration: {(time.time() - start_time_eval):.2f} sec")

                    # plot the prediction
                    print("Plot of last prediction (test set) and target:")

                # update model state_dict files on disk

                # update training parameters
                if train_eval_dict['runs'][test_num]['best_train_loss'] > train_loss:
                    train_eval_dict['runs'][test_num]['best_train_loss'] = train_loss
                    train_eval_dict['runs'][test_num]['best_train_epoch'] = epoch
                # update validation parameters
                if train_eval_dict['runs'][test_num]['best_val_loss'] > val_loss:
                    train_eval_dict['runs'][test_num]['best_val_loss'] = val_loss
                    train_eval_dict['runs'][test_num]['best_val_epoch'] = epoch
                    train_eval_dict['runs'][test_num]['conf_matr_list'] = conf_matr_list
                    epochs_since_last_improvement = 0
                # update global parameters
                if train_eval_dict['global_best_val_loss'] > val_loss:
                    train_eval_dict['global_best_val_loss'] = val_loss
                    train_eval_dict['global_best_val_test_num'] = test_num
                    train_eval_dict['global_best_val_epoch'] = epoch
                    # save model to drive
                    torch.save(model.state_dict(), best_val_model_fpath)

                if mopa_dict['es'] and epochs_since_last_improvement >= mopa_dict['esp']:
                    print(f" Early stopping after {mopa_dict['esp']} epochs.")
                    break

                # Decays the learning rate of each parameter group by gamma every step_size epochs
                # https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.StepLR.html
                if scheduler is not None:
                    scheduler.step()

            # finish up fold
            stop_time_fold = time.time()
            train_eval_dict['runs'][test_num]['duration_total'] = stop_time_fold - start_time_fold
            print(f"total duration of fold #{test_num}: {train_eval_dict['runs'][test_num]['duration_total']:.2f} sec")
            print(
                f"best val loss: {train_eval_dict['runs'][test_num]['best_val_loss']:.2f} in epoch: {train_eval_dict['runs'][test_num]['best_val_epoch']}")
            print()

            # free up memory
            del model
            torch.cuda.empty_cache()

        # ----------------------------------------------------------------------------------------------------------------------
        # Final testing
        # ----------------------------------------------------------------------------------------------------------------------

        mopa_dict = train_eval_dict['runs'][train_eval_dict['global_best_val_test_num']]['mopa_dict']

        final_test_dict = {
            'test_loss_list': [],
            'test_average_loss': math.inf,
            'test_conf_matr': tuple()
        }

        testset = DsI2CVB(I2CVB_basedir=ds_dict['fp'], include_patients=test_patients_li, mr_sequence="T2W",
                         transform=test_compose, num_of_surrouding_imgs=1,
                         include_classes=ds_dict['ice'], target_one_hot=target_one_hot,
                         samples_must_include_classes=ds_dict['mic'])

        test_loader = DataLoader(testset, batch_size=mopa_dict['bs'], shuffle=True)

        model = UNetSlim(3 + 3, num_classes, bilinear=True)
        model.load_state_dict(torch.load(best_val_model_fpath))
        # load model from file
        # net.load_state_dict(torch.load(run_autoenc_conf.best_val_model_name))
        model = model.to(mopa_dict['dev'])

        # start final test evaluation
        print()
        print("Start final test evaluation:")
        start_time_eval = time.time()
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            # conf_matr_list contains all the confusion matrices for the val_loader
            conf_matr_list = []
            for b_num, batch in enumerate(test_loader):
                inputs, targets, _ = batch
                inputs, targets = inputs.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                # add the prostate segmentation to the input
                inputs_e = torch.cat((inputs, targets[:, 2].unsqueeze(dim=1), targets[:, 3].unsqueeze(dim=1),
                                      targets[:, 4].unsqueeze(dim=1)), dim=1)
                targets = targets[:, :2]
                inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                inputs_e, targets = inputs_e.to(mopa_dict['dev']), targets.to(mopa_dict['dev'])

                output = model(inputs_e)
                # criterion
                loss = loss_fn(output, targets)

                final_test_dict['test_loss_list'].append(loss.cpu().detach().numpy())
                # add to the complete loss for the dataloader, multiply w/ the batchsize inputs.shape[0] in order to
                # later calculate the average

                test_loss += loss.data.item() * inputs.shape[0]

                # dimensions are N x C x H x W
                output = normalization(output)
                # determine the winner class
                output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

                conf_matr_list.append(
                    confusion_matrix(output_one_hot, targets, num_classes=num_classes, batch=True,
                                     target_one_hot=target_one_hot))

                # Progress info
                sys.stdout.write(
                    f"Test progress: {(((b_num + 1) * inputs.shape[0]) / len(testset)) * 100:.2f} % "
                    f"Test loss (batch): {loss.data.item():.3f} \r")
                sys.stdout.flush()

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
            test_loss /= len(testset)
            print(f"test Loss: {test_loss:.2f}  | Duration: {(time.time() - start_time_eval):.2f} sec")

            final_test_dict['test_average_loss'] = test_loss
            final_test_dict['test_conf_matr'] = conf_matr_list

            train_eval_dict['final_test_dict'] = final_test_dict

        # save training results
        fpath = os.path.join(run_path, model_fname)
        with open(fpath, 'wb') as f:
            pickle.dump(train_eval_dict, f, pickle.HIGHEST_PROTOCOL)