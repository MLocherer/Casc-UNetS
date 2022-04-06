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
import os
import math
import time
import pickle
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import OrderedDict, namedtuple
from itertools import product
import pandas as pd


class RunBuilder():
    r"""
    source: https://deeplizard.com/learn/video/NSKghk0pcco
    """

    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


class Run(object):
    def __init__(self, learning_rate: float, batch_size: int, epochs: int, loss_fun: str, normalisation: str,
                 include_class_labels: list, device: str, run_name=None, save_model=True, save_dir=None):
        r"""
        Constructor
        Args:
            learning_rate: model learning rate
            batch_size: batch size of train and data loader
            epochs: number of epochs
            loss_fun: name of the loss function
            device: device model runs on
            include_class_labels: list of class labels that are evaluated
            run_name: name that will be given to the model to identify it otherwise, the name is made up by the loss_fun
            and epochs
            save_model: if True the model state_dict will be saved to disk. This is true for the best training- and
            validation loss
            save_dir: directory in which the model state_dict, as well as the Run class object is stored
        """
        # basic info
        assert normalisation in ["sigmoid", "softmax", "identity"], "unknown normalisation specified, allowed types " \
                                                                    "are: sigmoid, softmax, or identity "
        self.normalisation = normalisation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fun = loss_fun
        self.include_class_labels = include_class_labels
        self.device = device
        self.save_model = save_model
        self.train_loss = []
        self.val_loss = []
        self.best_train_loss = math.inf
        self.best_train_loss_epoch = -1
        self.best_val_loss = math.inf
        self.best_val_loss_epoch = -1
        # run name
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = f"{loss_fun}_{epochs}"
        self.save_dir = os.path.join(save_dir, self.run_name)
        # timing info
        self.start_time = 0.0
        self.stop_time = 0.0
        self.duration = 0.0
        # handling the model parameters
        if self.save_model:
            if self.save_dir is not None:
                self.best_train_model_name = os.path.join(self.save_dir, "best_train_model.pt")
                self.best_val_model_name = os.path.join(self.save_dir, "best_val_model.pt")
            else:
                raise Exception(f"If the model has to be saved (save_model=True), a dir has to be specified in "
                                f"save_model_dir")
        else:
            self.best_train_model_name = ""
            self.best_val_model_name = ""

        # confusion matrix is dict of form {0: {tp: , fp: , tn:, fn:}, 1: {...}, ..., epochs - 1: {...}}
        self.confusion_matrix = {}
        # class weights
        self.class_weights = None

    @classmethod
    def load_from_disk(cls, file_path: str):
        r"""
        Loads the an Run object from disk.
        Args:
            file_path:

        Returns:

        """
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise Exception(f"file path file_path = {file_path} to load object from does not exist.")

    def __repr__(self):
        class_dict = {"learning_rate": self.learning_rate, "batch_size": self.batch_size, "epochs": self.epochs,
                      "loss_fun": self.loss_fun, "normalisation": self.normalisation,
                      "include_class_labels": self.include_class_labels,
                      "device": self.device, "best_train_loss": self.best_train_loss,
                      "best_train_loss_epoch": self.best_train_loss_epoch, "best_val_loss": self.best_val_loss,
                      "best_val_loss_epoch": self.best_val_loss_epoch, "run_name": self.run_name,
                      "start_time": self.start_time, "stop_time": self.stop_time, "duration": self.duration,
                      "save_name": self.save_dir, "best_train_model_name": self.best_train_model_name,
                      "best_val_model_name": self.best_val_model_name}
        return f"Run {class_dict}"

    def __str__(self):
        return f"Run object print:\n learning_rate = {self.learning_rate} \n batch_size = {self.batch_size} \n" \
               f" epochs = {self.epochs} \n loss_fun = {self.loss_fun} \n normalisation = {self.normalisation}" \
               f" class_weights = {self.class_weights} \n" \
               f" device = {self.device} \n run_name = {self.run_name} \n start_time = {self.start_time} \n" \
               f" stop_time = {self.stop_time} \n duration = {self.duration} \n" \
               f" best_train_model_name = {self.best_train_model_name} \n best_val_model_name = {self.best_val_model_name}"

    def start(self):
        r"""
        Starts the process to track time
        Returns:

        """
        self.start_time = time.time()
        return self.start_time

    def stop(self):
        r"""
        Stops the process to track time. Calculates duration.
        Returns:

        """
        self.stop_time = time.time()
        self.duration = self.stop_time - self.start_time
        return self.stop_time

    def update_model(self, current_epoch_avg_train_loss, current_epoch_avg_val_loss, current_epoch: int,
                     confusion_matrix: list, model_state_dict):
        r"""
        This function updates the model state_dict files based on the new loss values. If the loss has diminished, then
        the new model will be saved otherwise, it just updates the losses and the loss lists. Watch out: the loss values
        must be averages over the current epoch!
        Args:
            confusion_matrix: is a tuple or list of tp, fp, tn, fn of the current evaluation run
            current_epoch_avg_train_loss:
            current_epoch_avg_val_loss:
            model_state_dict:

        Returns:

        """
        # update train list
        self.train_loss.append(current_epoch_avg_train_loss)
        # check if model has to be saved and new best model is achieved
        if self.best_train_loss > current_epoch_avg_train_loss:
            if self.save_model:
                if os.path.isfile(self.best_train_model_name):
                    # delete old model files
                    os.remove(self.best_train_model_name)
                # save new ones
                torch.save(model_state_dict, self.best_train_model_name)
            # update loss
            self.best_train_loss = current_epoch_avg_train_loss
            self.best_train_loss_epoch = current_epoch

        # update val list
        self.val_loss.append(current_epoch_avg_val_loss)
        # check if model has to be saved and new best model is achieved
        if self.best_val_loss > current_epoch_avg_val_loss:
            if self.save_model:
                if os.path.isfile(self.best_val_model_name):
                    # delete old model files
                    os.remove(self.best_val_model_name)
                # save new ones
                torch.save(model_state_dict, self.best_val_model_name)
            self.best_val_loss = current_epoch_avg_val_loss
            self.best_val_loss_epoch = current_epoch

        # add confusion matrix
        self.confusion_matrix[current_epoch] = {'tp': confusion_matrix[0], 'fp': confusion_matrix[1],
                                                'tn': confusion_matrix[2], 'fn': confusion_matrix[3]}

    def save_to_disk(self):
        r"""
        Saves the current state of the object to disk, to the folder specified under save_dir in the __init__() method.
        Returns:

        """
        save_name = os.path.join(self.save_dir, f"{self.run_name}.pickle")
        with open(save_name, 'wb') as f:
            pickle.dump(self, f)

    def print_and_plot(self):
        r"""
        Plats the loss functions that are stored within a run object
        Args:
        Returns:
            None
        """
        # print
        print(self.__str__())
        if len(self.train_loss) != 0 and len(self.val_loss) != 0:
            print(f"best training loss = {self.best_train_loss} in epoch: {self.best_train_loss_epoch} \n best val "
                  f"loss = {self.best_val_loss} in epoch: {self.best_val_loss_epoch}")
            # plot
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(self.train_loss, color="tab:orange")
            ax1.set_title(f"train_loss len={len(self.train_loss)}")
            ax1.set_xlabel("iteration")
            ax1.set_ylabel("loss")
            ax1.grid()
            ax2.plot(self.val_loss, color="tab:blue")
            ax2.set_title(f"val_loss len={len(self.val_loss)}")
            ax2.set_xlabel("iteration")
            ax2.set_ylabel("loss")
            ax2.grid()
            # save figure
            save_name = os.path.join(self.save_dir, f"{self.run_name}_loss.png")
            plt.savefig(save_name)
            plt.tight_layout()
            plt.show()

        for epoch in range(len(self.confusion_matrix.keys())):
            print(f"epoch: {epoch}:")
            tabulate_conf_matr(self.confusion_matrix[epoch], self.include_class_labels)
            print()


def tabulate_conf_matr(conf_matr, labels):
    r"""
    prints out a dict or list of tp, fp, tn, fn in a nice manner
    Args:
        conf_matr: dict or list or tuple of numpy arrays
        labels: names for the labels in each column of the same

    Returns:
    """
    if type(conf_matr) == dict:
        tp = conf_matr['tp']
        fp = conf_matr['fp']
        tn = conf_matr['tn']
        fn = conf_matr['fn']
    else:
        tp = conf_matr[0]
        fp = conf_matr[1]
        tn = conf_matr[2]
        fn = conf_matr[3]

    # calculate metrics
    recall = tp / (tp + fn)
    # precision*
    precision = tp / (tp + fp)
    # dice / f1*
    dice = 2 * tp / (2 * tp + fp + fn)
    # iou*
    iou = tp / (tp + fp + fn)

    # create lists for table
    metric_l = labels.copy()
    metric_l.insert(0, 'metric')
    tp_l = tp.tolist()
    tp_l.insert(0, 'tp')
    fp_l = fp.tolist()
    fp_l.insert(0, 'fp')
    tn_l = tn.tolist()
    tn_l.insert(0, 'tn')
    fn_l = fn.tolist()
    fn_l.insert(0, 'fn')
    recall_l = recall.tolist()
    recall_l.insert(0, 'recall')
    precision_l = precision.tolist()
    precision_l.insert(0, 'precision')
    dice_l = dice.tolist()
    dice_l.insert(0, 'dice / f1')
    iou_l = iou.tolist()
    iou_l.insert(0, 'iou')

    table = [metric_l, tp_l, fp_l, tn_l, fn_l, recall_l, precision_l, dice_l, iou_l]

    print(tabulate(table, headers="firstrow", tablefmt="grid", floatfmt=".2f"))


def tabulate_train_eval_dict(train_eval_dict, labels, num_classes, device):
    r"""
    prints out a dict or list of tp, fp, tn, fn in a nice manner
    Args:
        conf_matr: dict or list or tuple of numpy arrays
        labels: names for the labels in each column of the same

    Returns:
    """
    num_folds = train_eval_dict['folds']

    # general evaluation over all folds
    recall_avg = torch.zeros(size=(num_folds, num_classes))
    precision_avg = torch.zeros(size=(num_folds, num_classes))
    iou_avg = torch.zeros(size=(num_folds, num_classes))
    dice_avg = torch.zeros(size=(num_folds, num_classes))

    for f in range(num_folds):
        tp_sum = torch.zeros(num_classes).to(device)
        fp_sum = torch.zeros(num_classes).to(device)
        tn_sum = torch.zeros(num_classes).to(device)
        fn_sum = torch.zeros(num_classes).to(device)

        # sum up over all batches
        for tp, fp, tn, fn in train_eval_dict['runs'][f]['conf_matr_list']:
            tp_sum += torch.sum(tp, dim=0)
            fp_sum += torch.sum(fp, dim=0)
            tn_sum += torch.sum(tn, dim=0)
            fn_sum += torch.sum(fn, dim=0)

        recall_avg[f] = tp_sum / (tp_sum + fn_sum)
        precision_avg[f] = tp_sum / (tp_sum + fp_sum)
        iou_avg[f] = tp_sum / (tp_sum + fp_sum + fn_sum)
        dice_avg[f] = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum)

    # create lists for table
    metric_l = labels.copy()
    metric_l.insert(0, 'metric')
    recall_l = torch.mean(recall_avg, dim=0).tolist()
    recall_l.insert(0, 'recall')
    recall_l_s = torch.std(recall_avg, dim=0, unbiased=True).tolist()
    recall_l_s.insert(0, 'recall std')
    precision_l = torch.mean(precision_avg, dim=0).tolist()
    precision_l.insert(0, 'precision')
    precision_l_s = torch.std(precision_avg, dim=0, unbiased=True).tolist()
    precision_l_s.insert(0, 'precision std')
    dice_l = torch.mean(dice_avg, dim=0).tolist()
    dice_l.insert(0, 'dice / f1')
    dice_l_s = torch.std(dice_avg, dim=0, unbiased=True).tolist()
    dice_l_s.insert(0, 'dice / f1 std')
    iou_l = torch.mean(iou_avg, dim=0).tolist()
    iou_l.insert(0, 'iou')
    iou_l_s = torch.std(iou_avg, dim=0, unbiased=True).tolist()
    iou_l_s.insert(0, 'iou std')

    table = [metric_l, recall_l, recall_l_s, precision_l, precision_l_s, dice_l, dice_l_s, iou_l, iou_l_s]

    print(tabulate(table, headers="firstrow", tablefmt="grid", floatfmt=".2f"))


def tabulate_5cv_results(conf_matr_5cv, imgnames_5cv, include_classes, num_classes, show_violin_plot=False):
    measures_dict = {}

    for ds_name_key in conf_matr_5cv:

        # slice based evaluation
        measures_dict[ds_name_key] = {}

        print(ds_name_key)
        # calculate min / max values for recall, precision, f1 and iou over all folds and batches as average and
        # for the individual biopsy regions
        # individual values for:
        # now biopsy region based evaluation
        # apex, middle, base
        num_slices = 0

        for batch_c in conf_matr_5cv[ds_name_key]:
            num_slices += len(batch_c[0])

        measures_dict[ds_name_key]['tp'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.int)
        measures_dict[ds_name_key]['fp'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.int)
        measures_dict[ds_name_key]['tn'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.int)
        measures_dict[ds_name_key]['fn'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.int)
        measures_dict[ds_name_key]['tpr'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict[ds_name_key]['ppv'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict[ds_name_key]['dsc'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict[ds_name_key]['iou'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)

        # biopsy indices
        measures_dict[ds_name_key]['indices_apex'] = []
        measures_dict[ds_name_key]['indices_middle'] = []
        measures_dict[ds_name_key]['indices_base'] = []

        i_start = 0
        for b_n, batch_c in enumerate(conf_matr_5cv[ds_name_key]):
            # average values over all biopsy regions
            i_stop = i_start + len(batch_c[0])
            tpb, fpb, tnb, fnb = batch_c
            measures_dict[ds_name_key]['tp'][i_start:i_stop] = tpb
            measures_dict[ds_name_key]['fp'][i_start:i_stop] = fpb
            measures_dict[ds_name_key]['tn'][i_start:i_stop] = tnb
            measures_dict[ds_name_key]['fn'][i_start:i_stop] = fnb
            measures_dict[ds_name_key]['tpr'][i_start:i_stop] = tpb / (tpb + fnb)
            measures_dict[ds_name_key]['ppv'][i_start:i_stop] = tpb / (tpb + fpb)
            measures_dict[ds_name_key]['dsc'][i_start:i_stop] = 2 * tpb / (2 * tpb + fpb + fnb)
            measures_dict[ds_name_key]['iou'][i_start:i_stop] = tpb / (tpb + fpb + fnb)
            # values over individual biopsy region
            # apex
            measures_dict[ds_name_key]['indices_apex'].extend(
                [(i + i_start) for i, value in enumerate(imgnames_5cv[ds_name_key][b_n]['biopsy_region']) if
                 value == 'apex'])
            measures_dict[ds_name_key]['indices_middle'].extend(
                [(i + i_start) for i, value in enumerate(imgnames_5cv[ds_name_key][b_n]['biopsy_region']) if
                 value == 'middle'])
            measures_dict[ds_name_key]['indices_base'].extend(
                [(i + i_start) for i, value in enumerate(imgnames_5cv[ds_name_key][b_n]['biopsy_region']) if
                 value == 'base'])
            i_start = i_stop

        """
        measures_dict[ds_name_key]['tpr'][torch.isnan(measures_dict[ds_name_key]['tpr'])] = 0.0
        measures_dict[ds_name_key]['ppv'][torch.isnan(measures_dict[ds_name_key]['ppv'])] = 0.0
        measures_dict[ds_name_key]['dsc'][torch.isnan(measures_dict[ds_name_key]['dsc'])] = 0.0
        measures_dict[ds_name_key]['iou'][torch.isnan(measures_dict[ds_name_key]['iou'])] = 0.0
        """
        df_data = {'patient': [],
                   'sample_id': [],
                   'biopsy_region': []}

        # add tp, fp, tn, fn, dscs for each class
        for class_idx in range(measures_dict[ds_name_key]['dsc'].shape[1]):
            if class_idx > 0:  # omit bg class
                df_data[f"tp{class_idx}"] = measures_dict[ds_name_key]['tp'][:, class_idx].tolist()
                df_data[f"fp{class_idx}"] = measures_dict[ds_name_key]['fp'][:, class_idx].tolist()
                df_data[f"tn{class_idx}"] = measures_dict[ds_name_key]['tn'][:, class_idx].tolist()
                df_data[f"fn{class_idx}"] = measures_dict[ds_name_key]['fn'][:, class_idx].tolist()
                df_data[f"dsc{class_idx}"] = measures_dict[ds_name_key]['dsc'][:, class_idx].tolist()

        for entry in imgnames_5cv[ds_name_key]:
            df_data['patient'].extend(entry['patient'])
            df_data['sample_id'].extend(entry['sample_id'].tolist())
            df_data['biopsy_region'].extend(entry['biopsy_region'])

        df = pd.DataFrame(data=df_data)

        df.to_csv(f"daten_{ds_name_key}.csv")

        measures_volume_dict = {}

        for patient in set(df_data['patient']):
            measures_volume_dict[patient] = {}
            for include_class in include_classes:
                measures_volume_dict[patient][f"complete{include_class}"] = 0
                for biopsy_region in set(df['biopsy_region']):
                    measures_volume_dict[patient][f"{biopsy_region}{include_class}"] = 0

        for patient in measures_volume_dict.keys():
            for include_class in include_classes:
                # complete region
                tp = df.loc[df['patient'] == patient][f"tp{include_class}"].sum()
                fp = df.loc[df['patient'] == patient][f"fp{include_class}"].sum()
                fn = df.loc[df['patient'] == patient][f"fn{include_class}"].sum()
                measures_volume_dict[patient][F"complete{include_class}"] = 2 * tp / (2 * tp + fp + fn)

                for biopsy_region in set(df['biopsy_region']):
                    tp = df.loc[(df['patient'] == patient) & (df['biopsy_region'] == biopsy_region)][
                        f"tp{include_class}"].sum()
                    fp = df.loc[(df['patient'] == patient) & (df['biopsy_region'] == biopsy_region)][
                        f"fp{include_class}"].sum()
                    fn = df.loc[(df['patient'] == patient) & (df['biopsy_region'] == biopsy_region)][
                        f"fn{include_class}"].sum()
                    measures_volume_dict[patient][f"{biopsy_region}{include_class}"] = 2 * tp / (2 * tp + fp + fn)

        df_volume = pd.DataFrame(data=measures_volume_dict)
        df_volume.to_csv(f"daten_volume_{ds_name_key}.csv")

        print("3D volume evaluation")
        print("mean\n", df_volume.mean(axis=1).round(decimals=3))
        print("std\n", df_volume.std(axis=1).round(decimals=3))

        for include_class in include_classes:
            print('class name: ', include_class)
            # create lists for table
            header_row = ['region', 'metric', 'min', 'max', 'mean', 'std']

            # There are nan values in the prediction tensors, they were generated from slices that do not have a cg / pz or
            # sometimes nothing is detected neither tp, fp, fn -> they must be removed otherwise wrong results
            # complete
            tpr = measures_dict[ds_name_key]['tpr'][:, include_class]
            tpr = tpr[~torch.isnan(tpr)]
            ppv = measures_dict[ds_name_key]['ppv'][:, include_class]
            ppv = ppv[~torch.isnan(ppv)]
            dsc = measures_dict[ds_name_key]['dsc'][:, include_class]
            dsc = dsc[~torch.isnan(dsc)]
            iou = measures_dict[ds_name_key]['iou'][:, include_class]
            iou = iou[~torch.isnan(iou)]

            # complete biopsy zones
            recall_l = ['complete', 'TPR', torch.min(tpr).tolist(),
                        torch.max(tpr).tolist(),
                        torch.mean(tpr).tolist(),
                        torch.std(tpr, unbiased=True).tolist()]
            precision_l = ['', 'PPV', torch.min(ppv).tolist(),
                           torch.max(ppv).tolist(),
                           torch.mean(ppv).tolist(),
                           torch.std(ppv, unbiased=True).tolist()]
            dice_l = ['', 'DSC', torch.min(dsc).tolist(),
                      torch.max(dsc).tolist(),
                      torch.mean(dsc).tolist(),
                      torch.std(dsc, unbiased=True).tolist()]
            iou_l = ['', 'IoU', torch.min(iou).tolist(),
                     torch.max(iou).tolist(),
                     torch.mean(iou).tolist(),
                     torch.std(iou, unbiased=True).tolist()]

            # apex
            tpr = measures_dict[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_apex'], include_class]
            tpr = tpr[~torch.isnan(tpr)]
            ppv = measures_dict[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_apex'], include_class]
            ppv = ppv[~torch.isnan(ppv)]
            dsc = measures_dict[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_apex'], include_class]
            dsc = dsc[~torch.isnan(dsc)]
            iou = measures_dict[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_apex'], include_class]
            iou = iou[~torch.isnan(iou)]

            recall_a_l = ['apex', 'TPR', torch.min(tpr).tolist(),
                          torch.max(tpr).tolist(),
                          torch.mean(tpr).tolist(),
                          torch.std(tpr, unbiased=True).tolist()]
            precision_a_l = ['', 'PPV', torch.min(ppv).tolist(),
                             torch.max(ppv).tolist(),
                             torch.mean(ppv).tolist(),
                             torch.std(ppv, unbiased=True).tolist()]
            dice_a_l = ['', 'DSC', torch.min(dsc).tolist(),
                        torch.max(dsc).tolist(),
                        torch.mean(dsc).tolist(),
                        torch.std(dsc, unbiased=True).tolist()]
            iou_a_l = ['', 'IoU', torch.min(iou).tolist(),
                       torch.max(iou).tolist(),
                       torch.mean(iou).tolist(),
                       torch.std(iou, unbiased=True).tolist()]

            # middle
            tpr = measures_dict[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_middle'], include_class]
            tpr = tpr[~torch.isnan(tpr)]
            ppv = measures_dict[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_middle'], include_class]
            ppv = ppv[~torch.isnan(ppv)]
            dsc = measures_dict[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_middle'], include_class]
            dsc = dsc[~torch.isnan(dsc)]
            iou = measures_dict[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_middle'], include_class]
            iou = iou[~torch.isnan(iou)]

            recall_m_l = ['middle', 'TPR', torch.min(tpr).tolist(),
                          torch.max(tpr).tolist(),
                          torch.mean(tpr).tolist(),
                          torch.std(tpr, unbiased=True).tolist()]
            precision_m_l = ['', 'PPV', torch.min(ppv).tolist(),
                             torch.max(ppv).tolist(),
                             torch.mean(ppv).tolist(),
                             torch.std(ppv, unbiased=True).tolist()]
            dice_m_l = ['', 'DSC', torch.min(dsc).tolist(),
                        torch.max(dsc).tolist(),
                        torch.mean(dsc).tolist(),
                        torch.std(dsc, unbiased=True).tolist()]
            iou_m_l = ['', 'IoU', torch.min(iou).tolist(),
                       torch.max(iou).tolist(),
                       torch.mean(iou).tolist(),
                       torch.std(iou, unbiased=True).tolist()]
            # base
            tpr = measures_dict[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_base'], include_class]
            tpr = tpr[~torch.isnan(tpr)]
            ppv = measures_dict[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_base'], include_class]
            ppv = ppv[~torch.isnan(ppv)]
            dsc = measures_dict[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_base'], include_class]
            dsc = dsc[~torch.isnan(dsc)]
            iou = measures_dict[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_base'], include_class]
            iou = iou[~torch.isnan(iou)]

            recall_b_l = ['base', 'TPR', torch.min(tpr).tolist(),
                          torch.max(tpr).tolist(),
                          torch.mean(tpr).tolist(),
                          torch.std(tpr, unbiased=True).tolist()]
            precision_b_l = ['', 'PPV', torch.min(ppv).tolist(),
                             torch.max(ppv).tolist(),
                             torch.mean(ppv).tolist(),
                             torch.std(ppv, unbiased=True).tolist()]
            dice_b_l = ['', 'DSC', torch.min(dsc).tolist(),
                        torch.max(dsc).tolist(),
                        torch.mean(dsc).tolist(),
                        torch.std(dsc, unbiased=True).tolist()]
            iou_b_l = ['', 'IoU', torch.min(iou).tolist(),
                       torch.max(iou).tolist(),
                       torch.mean(iou).tolist(),
                       torch.std(iou, unbiased=True).tolist()]

            table = [header_row, recall_l, precision_l, dice_l, iou_l, recall_a_l, precision_a_l, dice_a_l, iou_a_l,
                     recall_m_l,
                     precision_m_l, dice_m_l, iou_m_l, recall_b_l, precision_b_l, dice_b_l, iou_b_l]

            print(tabulate(table, headers="firstrow", tablefmt="latex", floatfmt=".3f"))

    if show_violin_plot:
        # dsc dict over all regions and classes
        regions = ['complete', 'apex', 'middle', 'base']
        for include_class in include_classes:
            d = {}
            for region in regions:
                d[region] = {}
                for ds_name_key in measures_dict:
                    if region == 'complete':
                        dsc = torch.squeeze(measures_dict[ds_name_key]['dsc'][:, include_class])
                        dsc = dsc[~torch.isnan(dsc)]
                        d[region][ds_name_key] = dsc.cpu().numpy()
                    else:
                        dsc = measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key][f"indices_{region}"], include_class]
                        dsc = dsc[~torch.isnan(dsc)]
                        d[region][ds_name_key] = dsc.cpu().numpy()

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 11.11))
            for tile, region in enumerate(d):
                r = tile // 2
                c = tile % 2

                plotdata = [d[region][ds] for ds in d[region].keys()]
                sns.violinplot(data=plotdata, cut=0, ax=axs[r, c], scale='count', palette="tab10")

                axs[r, c].set_title(region, style='italic', fontsize=24)

                # x axis
                axs[r, c].tick_params(axis='both', which='major', labelsize=18)
                axs[r, c].axes.grid(b=True, axis='y')

                axs[r, c].set_xticks([i for i in range(len(d[region].keys()))])
                axs[r, c].set_xticklabels([key for key in d[region].keys()])

                # axs[r, c].set_yticks([0.0, .2, .4, .6, .8, 1.0])

            for ax_r in range(axs.shape[0]):
                for ax_c in range(axs.shape[1]):
                    if ax_c:
                        # col 1
                        axs[ax_r, ax_c].set_yticklabels(axs[ax_r, ax_c].get_yticklabels(), visible=False)
                        axs[ax_r, ax_c].set_ylim(axs[ax_r, ax_c - 1].get_ylim())
                    else:
                        # col 0
                        axs[ax_r, ax_c].set_ylabel(f"DSC", fontsize=24)

            plt.tight_layout()
            plt.show()


def compare_models_5cv(conf_matr_5cv, conf_matr_5cv_combined, imgnames_5cv, include_classes, num_classes,
                       show_violin_plot=False):
    measures_dict = {}
    measures_dict_combined = {}
    for ds_name_key in conf_matr_5cv:
        measures_dict[ds_name_key] = {}
        measures_dict_combined[ds_name_key] = {}
        # calculate min / max values for recall, precision, f1 and iou over all folds and batches as average and
        # for the individual biopsy regions
        # individual values for:
        # now biopsy region based evaluation
        # apex, middle, base
        num_slices = 0

        for batch_c in conf_matr_5cv[ds_name_key]:
            num_slices += len(batch_c[0])

        measures_dict[ds_name_key]['tpr'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict[ds_name_key]['ppv'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict[ds_name_key]['dsc'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict[ds_name_key]['iou'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)

        measures_dict_combined[ds_name_key]['tpr'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict_combined[ds_name_key]['ppv'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict_combined[ds_name_key]['dsc'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)
        measures_dict_combined[ds_name_key]['iou'] = torch.zeros(size=(num_slices, num_classes), dtype=torch.float)

        # biopsy indices are the same for the combined so no need to store again
        measures_dict[ds_name_key]['indices_apex'] = []
        measures_dict[ds_name_key]['indices_middle'] = []
        measures_dict[ds_name_key]['indices_base'] = []

        i_start = 0
        for b_n, batch_c in enumerate(conf_matr_5cv[ds_name_key]):
            # average values over all biopsy regions
            i_stop = i_start + len(batch_c[0])
            tpb, fpb, tnb, fnb = batch_c
            # combined
            tpb_c, fpb_c, tnb_c, fnb_c = conf_matr_5cv_combined[ds_name_key][b_n]
            # original model
            measures_dict[ds_name_key]['tpr'][i_start:i_stop] = tpb / (tpb + fnb)
            measures_dict[ds_name_key]['ppv'][i_start:i_stop] = tpb / (fpb + fpb)
            measures_dict[ds_name_key]['dsc'][i_start:i_stop] = 2 * tpb / (2 * tpb + fpb + fnb)
            measures_dict[ds_name_key]['iou'][i_start:i_stop] = tpb / (tpb + fpb + fnb)
            # combined model results
            measures_dict_combined[ds_name_key]['tpr'][i_start:i_stop] = tpb_c / (tpb_c + fnb_c)
            measures_dict_combined[ds_name_key]['ppv'][i_start:i_stop] = tpb_c / (fpb_c + fpb_c)
            measures_dict_combined[ds_name_key]['dsc'][i_start:i_stop] = 2 * tpb_c / (2 * tpb_c + fpb_c + fnb_c)
            measures_dict_combined[ds_name_key]['iou'][i_start:i_stop] = tpb_c / (tpb_c + fpb_c + fnb_c)
            # values over individual biopsy region
            # apex
            measures_dict[ds_name_key]['indices_apex'].extend(
                [(i + i_start) for i, value in enumerate(imgnames_5cv[ds_name_key][b_n]['biopsy_region']) if
                 value == 'apex'])
            measures_dict[ds_name_key]['indices_middle'].extend(
                [(i + i_start) for i, value in enumerate(imgnames_5cv[ds_name_key][b_n]['biopsy_region']) if
                 value == 'middle'])
            measures_dict[ds_name_key]['indices_base'].extend(
                [(i + i_start) for i, value in enumerate(imgnames_5cv[ds_name_key][b_n]['biopsy_region']) if
                 value == 'base'])
            i_start = i_stop

        # filter out nan values in precision
        measures_dict[ds_name_key]['ppv'][torch.isnan(measures_dict[ds_name_key]['ppv'])] = 0.0
        measures_dict_combined[ds_name_key]['ppv'][torch.isnan(measures_dict_combined[ds_name_key]['ppv'])] = 0.0

        # create lists for table
        header_row = ['region', 'metric', 'min', 'max', 'mean', 'std']
        # complete biopsy zones
        recall_l = ['complete', 'TPR', (
                torch.min(measures_dict_combined[ds_name_key]['tpr'][:, include_classes]) - torch.min(
            measures_dict[ds_name_key]['tpr'][:, include_classes])).tolist(),
                    (torch.max(measures_dict_combined[ds_name_key]['tpr'][:, include_classes]) - torch.max(
                        measures_dict[ds_name_key]['tpr'][:, include_classes])).tolist(),
                    (torch.mean(measures_dict_combined[ds_name_key]['tpr'][:, include_classes]) - torch.mean(
                        measures_dict[ds_name_key]['tpr'][:, include_classes])).tolist(),
                    (torch.std(measures_dict_combined[ds_name_key]['tpr'][:, include_classes]) - torch.std(
                        measures_dict[ds_name_key]['tpr'][:, include_classes])).tolist()]
        precision_l = ['', 'PPV', (
                torch.min(measures_dict_combined[ds_name_key]['ppv'][:, include_classes]) - torch.min(
            measures_dict[ds_name_key]['ppv'][:, include_classes])).tolist(),
                       (torch.max(measures_dict_combined[ds_name_key]['ppv'][:, include_classes]) - torch.max(
                           measures_dict[ds_name_key]['ppv'][:, include_classes])).tolist(),
                       (torch.mean(measures_dict_combined[ds_name_key]['ppv'][:, include_classes]) - torch.mean(
                           measures_dict[ds_name_key]['ppv'][:, include_classes])).tolist(),
                       (torch.std(measures_dict_combined[ds_name_key]['ppv'][:, include_classes]) - torch.std(
                           measures_dict[ds_name_key]['ppv'][:, include_classes])).tolist()]
        dice_l = ['', 'DSC', (
                torch.min(measures_dict_combined[ds_name_key]['dsc'][:, include_classes]) - torch.min(
            measures_dict[ds_name_key]['dsc'][:, include_classes])).tolist(),
                  (torch.max(measures_dict_combined[ds_name_key]['dsc'][:, include_classes]) - torch.max(
                      measures_dict[ds_name_key]['dsc'][:, include_classes])).tolist(),
                  (torch.mean(measures_dict_combined[ds_name_key]['dsc'][:, include_classes]) - torch.mean(
                      measures_dict[ds_name_key]['dsc'][:, include_classes])).tolist(),
                  (torch.std(measures_dict_combined[ds_name_key]['dsc'][:, include_classes]) - torch.std(
                      measures_dict[ds_name_key]['dsc'][:, include_classes])).tolist()]
        iou_l = ['', 'IoU', (
                torch.min(measures_dict_combined[ds_name_key]['iou'][:, include_classes]) - torch.min(
            measures_dict[ds_name_key]['iou'][:, include_classes])).tolist(),
                 (torch.max(measures_dict_combined[ds_name_key]['iou'][:, include_classes]) - torch.max(
                     measures_dict[ds_name_key]['iou'][:, include_classes])).tolist(),
                 (torch.mean(measures_dict_combined[ds_name_key]['iou'][:, include_classes]) - torch.mean(
                     measures_dict[ds_name_key]['iou'][:, include_classes])).tolist(),
                 (torch.std(measures_dict_combined[ds_name_key]['iou'][:, include_classes]) - torch.std(
                     measures_dict[ds_name_key]['iou'][:, include_classes])).tolist()]
        # apex
        recall_a_l = ['apex', 'TPR', (torch.min(
            measures_dict_combined[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_apex'], 1]) - torch.min(
            measures_dict[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_apex'], 1])).tolist(),
                      (torch.max(measures_dict_combined[ds_name_key]['tpr'][
                                     measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.max(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                      (torch.mean(measures_dict_combined[ds_name_key]['tpr'][
                                      measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.mean(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                      (torch.std(measures_dict_combined[ds_name_key]['tpr'][
                                     measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.std(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist()]

        precision_a_l = ['', 'PPV', (torch.min(
            measures_dict_combined[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_apex'], 1]) - torch.min(
            measures_dict[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_apex'], 1])).tolist(),
                         (torch.max(measures_dict_combined[ds_name_key]['ppv'][
                                        measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.max(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                         (torch.mean(measures_dict_combined[ds_name_key]['ppv'][
                                         measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.mean(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                         (torch.std(measures_dict_combined[ds_name_key]['ppv'][
                                        measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.std(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist()]
        dice_a_l = ['', 'DSC', (torch.min(
            measures_dict_combined[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_apex'], 1]) - torch.min(
            measures_dict[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_apex'], 1])).tolist(),
                    (torch.max(measures_dict_combined[ds_name_key]['dsc'][
                                   measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.max(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                    (torch.mean(measures_dict_combined[ds_name_key]['dsc'][
                                    measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.mean(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                    (torch.std(measures_dict_combined[ds_name_key]['dsc'][
                                   measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.std(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist()]
        iou_a_l = ['', 'IoU', (torch.min(
            measures_dict_combined[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_apex'], 1]) - torch.min(
            measures_dict[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_apex'], 1])).tolist(),
                   (torch.max(measures_dict_combined[ds_name_key]['iou'][
                                  measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.max(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                   (torch.mean(measures_dict_combined[ds_name_key]['iou'][
                                   measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.mean(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist(),
                   (torch.std(measures_dict_combined[ds_name_key]['iou'][
                                  measures_dict[ds_name_key]['indices_apex'], include_classes]) - torch.std(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_apex'], include_classes])).tolist()]
        # middle
        recall_m_l = ['middle', 'TPR', (torch.min(
            measures_dict_combined[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_middle'], 1]) - torch.min(
            measures_dict[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_middle'], 1])).tolist(),
                      (torch.max(measures_dict_combined[ds_name_key]['tpr'][
                                     measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.max(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                      (torch.mean(measures_dict_combined[ds_name_key]['tpr'][
                                      measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.mean(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                      (torch.std(measures_dict_combined[ds_name_key]['tpr'][
                                     measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.std(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist()]
        precision_m_l = ['', 'PPV', (torch.min(
            measures_dict_combined[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_middle'], 1]) - torch.min(
            measures_dict[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_middle'], 1])).tolist(),
                         (torch.max(measures_dict_combined[ds_name_key]['ppv'][
                                        measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.max(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                         (torch.mean(measures_dict_combined[ds_name_key]['ppv'][
                                         measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.mean(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                         (torch.std(measures_dict_combined[ds_name_key]['ppv'][
                                        measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.std(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist()]
        dice_m_l = ['', 'DSC', (torch.min(
            measures_dict_combined[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_middle'], 1]) - torch.min(
            measures_dict[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_middle'], 1])).tolist(),
                    (torch.max(measures_dict_combined[ds_name_key]['dsc'][
                                   measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.max(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                    (torch.mean(measures_dict_combined[ds_name_key]['dsc'][
                                    measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.mean(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                    (torch.std(measures_dict_combined[ds_name_key]['dsc'][
                                   measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.std(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist()]
        iou_m_l = ['', 'IoU', (torch.min(
            measures_dict_combined[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_middle'], 1]) - torch.min(
            measures_dict[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_middle'], 1])).tolist(),
                   (torch.max(measures_dict_combined[ds_name_key]['iou'][
                                  measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.max(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                   (torch.mean(measures_dict_combined[ds_name_key]['iou'][
                                   measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.mean(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist(),
                   (torch.std(measures_dict_combined[ds_name_key]['iou'][
                                  measures_dict[ds_name_key]['indices_middle'], include_classes]) - torch.std(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_middle'], include_classes])).tolist()]
        # base
        recall_b_l = ['base', 'TPR', (torch.min(
            measures_dict_combined[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_base'], 1]) - torch.min(
            measures_dict[ds_name_key]['tpr'][measures_dict[ds_name_key]['indices_base'], 1])).tolist(),
                      (torch.max(measures_dict_combined[ds_name_key]['tpr'][
                                     measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.max(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                      (torch.mean(measures_dict_combined[ds_name_key]['tpr'][
                                      measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.mean(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                      (torch.std(measures_dict_combined[ds_name_key]['tpr'][
                                     measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.std(
                          measures_dict[ds_name_key]['tpr'][
                              measures_dict[ds_name_key]['indices_base'], include_classes])).tolist()]
        precision_b_l = ['', 'PPV', (torch.min(
            measures_dict_combined[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_base'], 1]) - torch.min(
            measures_dict[ds_name_key]['ppv'][measures_dict[ds_name_key]['indices_base'], 1])).tolist(),
                         (torch.max(measures_dict_combined[ds_name_key]['ppv'][
                                        measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.max(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                         (torch.mean(measures_dict_combined[ds_name_key]['ppv'][
                                         measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.mean(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                         (torch.std(measures_dict_combined[ds_name_key]['ppv'][
                                        measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.std(
                             measures_dict[ds_name_key]['ppv'][
                                 measures_dict[ds_name_key]['indices_base'], include_classes])).tolist()]
        indi = torch.mean(measures_dict_combined[ds_name_key]['dsc'][
                              measures_dict[ds_name_key]['indices_base'], include_classes])
        comb = torch.mean(
            measures_dict[ds_name_key]['dsc'][
                measures_dict[ds_name_key]['indices_base'], include_classes])
        delta = comb - indi
        dice_b_l = ['', 'DSC', (torch.min(
            measures_dict_combined[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_base'], 1]) - torch.min(
            measures_dict[ds_name_key]['dsc'][measures_dict[ds_name_key]['indices_base'], 1])).tolist(),
                    (torch.max(measures_dict_combined[ds_name_key]['dsc'][
                                   measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.max(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                    (torch.mean(measures_dict_combined[ds_name_key]['dsc'][
                                    measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.mean(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                    (torch.std(measures_dict_combined[ds_name_key]['dsc'][
                                   measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.std(
                        measures_dict[ds_name_key]['dsc'][
                            measures_dict[ds_name_key]['indices_base'], include_classes])).tolist()]
        iou_b_l = ['', 'IoU', (torch.min(
            measures_dict_combined[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_base'], 1]) - torch.min(
            measures_dict[ds_name_key]['iou'][measures_dict[ds_name_key]['indices_base'], 1])).tolist(),
                   (torch.max(measures_dict_combined[ds_name_key]['iou'][
                                  measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.max(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                   (torch.mean(measures_dict_combined[ds_name_key]['iou'][
                                   measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.mean(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_base'], include_classes])).tolist(),
                   (torch.std(measures_dict_combined[ds_name_key]['iou'][
                                  measures_dict[ds_name_key]['indices_base'], include_classes]) - torch.std(
                       measures_dict[ds_name_key]['iou'][
                           measures_dict[ds_name_key]['indices_base'], include_classes])).tolist()]

        table = [header_row, recall_l, precision_l, dice_l, iou_l, recall_a_l, precision_a_l, dice_a_l, iou_a_l,
                 recall_m_l,
                 precision_m_l, dice_m_l, iou_m_l, recall_b_l, precision_b_l, dice_b_l, iou_b_l]

        print(tabulate(table, headers="firstrow", tablefmt="latex", floatfmt=".3f"))

    if show_violin_plot:
        # dsc dict over all regions and classes
        regions = ['complete', 'apex', 'middle', 'base']

        d = {}
        for region in regions:
            d[region] = {}
            for ds_name_key in measures_dict:
                if region == 'complete':
                    d[region][ds_name_key] = torch.tensor(100) * (measures_dict_combined[ds_name_key]['dsc'][:,
                                                                  include_classes].cpu().numpy().squeeze() - \
                                                                  measures_dict[ds_name_key]['dsc'][:,
                                                                  include_classes].cpu().numpy().squeeze())
                else:
                    d[region][ds_name_key] = torch.tensor(100) * (measures_dict_combined[ds_name_key]['dsc'][
                                                                      measures_dict[ds_name_key][
                                                                          f"indices_{region}"], include_classes].cpu().numpy() - \
                                                                  measures_dict[ds_name_key]['dsc'][
                                                                      measures_dict[ds_name_key][
                                                                          f"indices_{region}"], include_classes].cpu().numpy())

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 11.11))
        for tile, region in enumerate(d):
            r = tile // 2
            c = tile % 2

            plotdata = [d[region][ds] for ds in d[region].keys()]
            # sns.violinplot(data=plotdata, cut=0, ax=axs[r, c], scale='count', gridsize=200, palette="tab10")
            sns.boxplot(data=plotdata, ax=axs[r, c], palette="tab10")
            # sns.swarmplot(data=plotdata, ax=axs[r, c], color="k", alpha=0.5)

            axs[r, c].set_title(region, style='italic', fontsize=24)

            # x axis
            axs[r, c].tick_params(axis='both', which='major', labelsize=18)
            axs[r, c].axes.grid(b=True, axis='y')

            axs[r, c].set_xticks([i for i in range(len(d[region].keys()))])
            axs[r, c].set_xticklabels([key for key in d[region].keys()])

            # axs[r, c].set_yticks(np.linspace(-30, 30, 13, endpoint=True))

        for ax_r in range(axs.shape[0]):
            for ax_c in range(axs.shape[1]):
                if ax_c:
                    # col 1
                    axs[ax_r, ax_c].set_yticklabels(axs[ax_r, ax_c].get_yticklabels(), visible=False)
                    axs[ax_r, ax_c].set_ylim(axs[ax_r, ax_c - 1].get_ylim())
                else:
                    # col 0
                    axs[ax_r, ax_c].set_ylabel(r'Delta DSC [%]', fontsize=24)

        plt.tight_layout()
        plt.show()


def print_loss_list(train_eval_dict, plot_title: str = None, test_num: int = None, save_fig: bool = False,
                    save_name: str = None):
    # plot config
    title_fontsize = 16
    ax_fontsize = 14

    if test_num is None:
        global_best_val_test_num = train_eval_dict['global_best_val_test_num']
        best_mopa_dict = train_eval_dict['runs'][global_best_val_test_num]['mopa_dict']
        best_train_list = train_eval_dict['runs'][global_best_val_test_num]['train_loss_list']
        best_val_list = train_eval_dict['runs'][global_best_val_test_num]['val_loss_list']
        best_val_loss = train_eval_dict['runs'][global_best_val_test_num]['best_val_loss']
        best_val_epoch = train_eval_dict['runs'][global_best_val_test_num]['best_val_epoch']
    else:
        best_mopa_dict = train_eval_dict['runs'][test_num]['mopa_dict']
        best_train_list = train_eval_dict['runs'][test_num]['train_loss_list']
        best_val_list = train_eval_dict['runs'][test_num]['val_loss_list']
        best_val_loss = train_eval_dict['runs'][test_num]['best_val_loss']
        best_val_epoch = train_eval_dict['runs'][test_num]['best_val_epoch']

    test_loss_list = train_eval_dict['final_test_dict']['test_loss_list']
    test_average_loss = train_eval_dict['final_test_dict']['test_average_loss']

    plot_title = f"{plot_title}: lr = {best_mopa_dict['lr']}, bs = {best_mopa_dict['bs']}, val loss = {best_val_loss:.2f} in epoch = {best_val_epoch}"

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle(plot_title, fontsize=title_fontsize)

    ax1.plot(best_train_list, color="tab:red")
    ax1.set_title(f"train_loss len={len(best_train_list)}", fontsize=ax_fontsize)
    ax1.set_xlabel("iteration", fontsize=ax_fontsize)
    ax1.set_ylabel("loss", fontsize=ax_fontsize)
    ax1.grid()
    ax2.plot(best_val_list, color="tab:orange")
    ax2.set_title(f"val_loss len={len(best_val_list)}", fontsize=ax_fontsize)
    ax2.set_xlabel("iteration", fontsize=ax_fontsize)
    ax2.set_ylabel("loss", fontsize=ax_fontsize)
    ax2.grid()
    ax3.plot(test_loss_list, color="tab:blue")
    ax3.axhline(test_average_loss, color="tab:blue")
    ax3.set_title(
        f"test_loss len={len(test_loss_list)}, average loss = {test_average_loss:.2f} [{min(test_loss_list):.2f}, {max(test_loss_list):.2f}]",
        fontsize=ax_fontsize)
    ax3.set_xlabel("iteration", fontsize=ax_fontsize)
    ax3.set_ylabel("loss", fontsize=ax_fontsize)
    ax3.grid()

    # save figure
    if save_fig:
        save_name = os.path.join(save_name, f"loss_list_plot.tex")
        plt.savefig(save_name)
        # tikzplotlib.clean_figure()
        # tikzplotlib.save(save_name)

    plt.tight_layout()
    plt.show()


def get_best_val_epochs(train_eval_dict):
    best_val_epochs = {}
    best_val_loss = {}

    for run in train_eval_dict['runs']:
        mopa_dict = train_eval_dict['runs'][run]['mopa_dict']
        lr = mopa_dict['lr']

        if lr not in best_val_epochs:
            best_val_epochs[lr] = []
            best_val_loss[lr] = []

        best_val_epochs[lr].append(train_eval_dict['runs'][run]['best_val_epoch'])
        best_val_loss[lr].append(train_eval_dict['runs'][run]['best_val_loss'])

    table = [['lr', 'bs', '2', '4', '6', '8', '10', 'avg epoch / avg val loss', 'std epoch / std val loss']]
    table_row = []
    for lr in best_val_epochs:
        avg_val_epoch = np.mean(best_val_epochs[lr])
        std_val_epoch = np.std(best_val_epochs[lr])

        avg_val_loss = np.mean(best_val_loss[lr])
        std_val_loss = np.std(best_val_loss[lr])
        # first row
        table_row.append(lr)
        table_row.append('epoch')
        table_row.extend(best_val_epochs[lr])
        table_row.append(f"{avg_val_epoch:.0f}")
        table_row.append(f"{std_val_epoch:.0f}")
        table.append(table_row.copy())
        table_row[:] = []
        # second row
        table_row.append('')
        table_row.append('val loss')
        for loss in best_val_loss[lr]:
            table_row.append(f"{loss:.3f}")
        table_row.append(f"{avg_val_loss:.3f}")
        table_row.append(f"{std_val_loss:.3f}")
        table.append(table_row.copy())
        table_row[:] = []

    print(tabulate(table, tablefmt="latex", floatfmt=".3f"))

    return best_val_epochs, best_val_loss


if __name__ == '__main__':
    pass
