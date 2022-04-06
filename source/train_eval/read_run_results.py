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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Dataset
from source.pymodules import Compose, ToTensor,  Resize
from source.pymodules import DsI2CVB, calc_i2cvb_weights, plot_prediction_contour, plot_segmentation_contour
# Model
from source.pymodules import UNetSlim
# Loss
from source.pymodules import confusion_matrix, get_normalisation_fun, \
    get_loss_fun
from source.pymodules import tabulate_conf_matr
# train_utils
from source.pymodules import ds_path_dict


def print_loss_list(train_eval_dict, plot_title: str = None, save_fig: bool = False, save_name: str = None):
    # plot config
    title_fontsize = 16
    ax_fontsize = 14

    global_best_val_test_num = train_eval_dict['global_best_val_test_num']
    best_mopa_dict = train_eval_dict['runs'][global_best_val_test_num]['mopa_dict']
    best_train_list = train_eval_dict['runs'][global_best_val_test_num]['train_loss_list']
    best_val_list = train_eval_dict['runs'][global_best_val_test_num]['val_loss_list']
    best_val_loss = train_eval_dict['runs'][global_best_val_test_num]['best_val_loss']
    best_val_epoch = train_eval_dict['runs'][global_best_val_test_num]['best_val_epoch']
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
    ax3.set_title(f"test_loss len={len(test_loss_list)}, average loss = {test_average_loss:.2f} [{min(test_loss_list):.2f}, {max(test_loss_list):.2f}]", fontsize=ax_fontsize)
    ax3.set_xlabel("iteration", fontsize=ax_fontsize)
    ax3.set_ylabel("loss", fontsize=ax_fontsize)
    ax3.grid()
    # save figure
    if save_fig:
        save_name = os.path.join(save_name, f"loss_list_plot.png")
        plt.savefig(save_name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------------------
    # General settings
    # ----------------------------------------------------------------------------------------------------------------------
    run_path = '/run/path/where/models/are/stored/'
    pickle_fname = 'test-2021-09-17_19-28-42.pickle'

    pickle_fpath = os.path.join(run_path, pickle_fname)
    image_export_path = os.path.join('/path/where/images/shall/be/stored/', os.path.splitext(pickle_fname)[0])

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
    print_loss_list(train_eval_dict, 'hello')

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

    print(ds_dict['fp'])
    print('change to local filepath accordingly: ')
    # update path here
    # ----------------------------------------------------------------------------------------------------------------------
    # path settings
    # ----------------------------------------------------------------------------------------------------------------------
    ds_type = 'local'
    ds_he = 'clahe'
    ds_name_key = 'combined'
    ds_dict['fp'] = ds_path_dict[ds_type][ds_he][ds_name_key]
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
            # dimensions are N x C x H x W
            output = normalization(output)
            # output_one_hot = (output >= 0.5) * 1
            # determine the winner class
            output_one_hot = F.one_hot(torch.argmax(output, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)

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
