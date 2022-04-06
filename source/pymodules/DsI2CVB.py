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
sys.path.append("pymodules")
import os
import glob
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from skimage import img_as_ubyte, img_as_uint, img_as_float32
from natsort import natsorted
from DsTransformations import feature_scaling
import matplotlib.pyplot as plt
from source.pymodules import Compose, ToTensor, Resize, ds_path_dict


class DsI2CVB(Dataset):
    def __init__(self, I2CVB_basedir, include_patients, mr_sequence, transform=None, num_of_surrouding_imgs=0,
                 include_classes=None, target_one_hot=True,
                 samples_must_include_classes=None):
        r"""
        Args:
            I2CVB_basedir: folder above all patients
            include_patients: list of patients to include (from k-fold) all patients have 64 images
            mr_sequence: which mr sequence to choose
            transform: composition of transforms
            num_of_surrouding_imgs: if set to 3 use alkadi et al approach otherwise single image
            include_classes: which classes to include
            target_one_hot: one-hot-encoded target
            samples_must_include_classes: if this list is set to one or more class(es) then only the images and segmentation
        maps are added for which at least one class is present.
        """
        self.I2CVB_basedir = I2CVB_basedir
        self.include_patients = include_patients

        if include_classes is None:
            include_classes = ['bg', 'cap', 'pz', 'cg', 'prostate']

        if mr_sequence in ['ADC', 'DCE', 'DWI', 'MRSI', 'T2W']:
            self.mr_sequence = mr_sequence
        else:
            assert False, f"sequence {mr_sequence} invalid."

        self.transform = transform
        self.num_of_surrouding_imgs = num_of_surrouding_imgs

        possible_classes = ['bg', 'cap', 'pz', 'cg', 'prostate']

        # the class 'cg' is not evaluated although it is part of I2CVB dataset
        self.include_classes = []
        for i_class in include_classes:
            if i_class not in possible_classes:
                assert False, f"class {i_class} in {include_classes} invalid."
            else:
                self.include_classes.append(i_class)

        self.target_one_hot = target_one_hot

        include_only_class_list_2 = []
        if samples_must_include_classes is not None:
            # check if include_only_class_list is a subset of possible_include_classes
            for i_class in samples_must_include_classes:
                if i_class not in possible_classes:
                    assert False, f"class {i_class} invalid."
                else:
                    include_only_class_list_2.append(i_class)

        # data storage for all images and segmentations
        self.p_dict = {}

        # p_dict_index stores the indices of each patient sample
        # p_id: patient id, s_idx: sample index
        # create a mapping {0: {'p_id': 0, 's_idx': 0}, ..., 27: {'p_id': 2, 's': 34}, ...}
        self.p_dict_index = {}

        # create a mapping dict index is p_id, value is length of images in patient number p_id
        # is used for sequence generation to find corner cases
        self.p_dict_length = {}

        # continous counter over all elements in p_dict used for p_dict_index
        sample_id_counter = 0

        # now we iterate over all patients
        for patient_num, patient in enumerate(include_patients):
            self.p_dict[patient_num] = {}
            # images
            mri_dir = os.path.join(I2CVB_basedir, patient, mr_sequence, '*')
            # load the DICOM files
            slices = []
            flist = natsorted(glob.glob(mri_dir, recursive=False))
            for fname in flist:
                img = sitk.ReadImage(fname)
                arr = sitk.GetArrayFromImage(img)
                slices.append(arr)

            # pixel array (numpy array of patients)
            slices_pa = np.concatenate(slices, axis=0)

            # segmentation we do not yet create a background class
            seg_dir = os.path.join(I2CVB_basedir, patient, 'GT')
            seg_list = []
            for seg in self.include_classes:
                if seg != 'bg':
                    # we skip background class since it will be generated after the transforms are applied
                    seg_dir_p = os.path.join(seg_dir, seg, '*')
                    slices = []
                    flist = natsorted(glob.glob(seg_dir_p, recursive=False))
                    for fname in flist:
                        img = sitk.ReadImage(fname)
                        arr = sitk.GetArrayFromImage(img)
                        slices.append(arr)

                    # pixel array (numpy array of patients)
                    seg_list.append(
                        np.expand_dims(np.concatenate(slices, axis=0),
                                       axis=1))

            include_indices = list(range(seg_list[0].shape[0]))

            if include_only_class_list_2 is not None:
                # only include images for which the segmentation include_only_class is present.
                include_indices = []

                for img_idx in range(seg_list[0].shape[0]):
                    or_gate_classes = 0
                    for cls_x in include_only_class_list_2:
                        # find the index of the segmentation map in self.include_classes since this list contains the
                        # segmentation maps in the order they were added to seg_list
                        cls_idx = self.include_classes.index(cls_x) - 1

                        or_gate_classes += np.sum(seg_list[cls_idx][img_idx])
                    if or_gate_classes > 0:
                        include_indices.append(img_idx)

                # print(len(include_indices))

            self.p_dict[patient_num]['id'] = patient
            self.p_dict[patient_num]['img'] = img_as_float32(slices_pa[include_indices])
            # self.p_dict[patient_num]['img'] = img_as_uint(slices_pa[include_indices])
            self.p_dict[patient_num]['seg'] = img_as_ubyte(np.concatenate(seg_list, axis=1)[include_indices])

            # now fill up p_dict_index
            length_img_data = len(self.p_dict[patient_num]['img'])
            for sample_id in range(length_img_data):
                self.p_dict_index[sample_id_counter] = {'p_id': patient_num, 's_idx': sample_id}
                sample_id_counter += 1
            # now add new length to self.p_dict_length
            self.p_dict_length[patient_num] = length_img_data

    def __len__(self):
        r"""
        Returns the length of the dataset
        """
        return len(self.p_dict_index)

    def __getitem__(self, idx):
        # it is assumed that each patient has at least two sample images
        p_idx, sample_id = self.p_dict_index[idx]['p_id'], self.p_dict_index[idx]['s_idx']
        p_max_samples = self.p_dict_length[p_idx]
        # determine biopsy region according to
        biopsy_region_percentile = np.around(sample_id / p_max_samples, decimals=2)
        # assign to region apex (0 - 15th percentile), middle (16 - 84th percentile), base (85 - 100th percentile)
        if biopsy_region_percentile <= 0.15:
            biopsy_region = 'apex'
        elif biopsy_region_percentile > 0.15 and biopsy_region_percentile <= 0.84:
            biopsy_region = 'middle'
        elif biopsy_region_percentile > 0.84:
            biopsy_region = 'base'
        else:
            biopsy_region = f'not properly assigned {biopsy_region_percentile}'
        # dict with patient id and sample id to be able to assign correct patients and predictions
        identifier = {'patient': self.p_dict[p_idx]['id'], 'sample_id': sample_id, 'biopsy_region': biopsy_region}

        # sequence generation
        # find out at which index the current index is:
        current_img = feature_scaling(self.p_dict[p_idx]['img'][sample_id])
        target = self.p_dict[p_idx]['seg'][sample_id] / 255

        if self.num_of_surrouding_imgs == 0:
            sample_img = current_img
        elif self.num_of_surrouding_imgs == 1:
            # corner cases
            # sample_id == 0 or last element
            if sample_id == 0:
                # first image
                previous_img = feature_scaling(np.copy(current_img))
                next_img = feature_scaling(self.p_dict[p_idx]['img'][sample_id + 1])
            elif sample_id == p_max_samples - 1:
                # last image
                previous_img = feature_scaling(self.p_dict[p_idx]['img'][sample_id - 1])
                next_img = feature_scaling(np.copy(current_img))
            else:
                # image in b/W
                previous_img = feature_scaling(self.p_dict[p_idx]['img'][sample_id - 1])
                next_img = feature_scaling(self.p_dict[p_idx]['img'][sample_id + 1])

            # create array of shape H x W x C this is necessary for to TF.ToTensor()
            sample_img = np.concatenate((np.expand_dims(previous_img, axis=2),
                                         np.expand_dims(current_img, axis=2),
                                         np.expand_dims(next_img, axis=2)), axis=2)

        if self.transform:
            # we have to at least call the ToTensor transform!
            sample_img, target = self.transform(sample_img, target)
            if 'bg' in self.include_classes:
                # create background class label
                bg = (torch.sum(target, dim=0, dtype=target[0].dtype) == 0) * 1
                target = torch.cat((torch.unsqueeze(bg, dim=0), target), dim=0)

        if not self.target_one_hot:
            if torch.is_tensor(target):
                target = torch.argmax(target, dim=0)
            else:
                target = np.argmax(target, axis=0)

        return sample_img, target, identifier


def get_I2CVB_dataset(I2CVB_basedir):
    r"""
    returns a list with all patient folder names of the I2CVB dataset.
    Args:
        I2CVB_basedir: basepath above patients

    Returns:
        list of patient folder names
    """
    I2CVB_basedir = os.path.join(I2CVB_basedir, '*')
    patients = []

    for fdir in glob.glob(I2CVB_basedir):
        if os.path.isdir(fdir):
            patients.append(os.path.basename(fdir))
    return patients


def calc_i2cvb_weights(I2CVB_dataset, include_classes, target_one_hot=False):
    r"""
    returns an array with weights assosiated with each class in I2CVB_dataset
    Args:
        target_one_hot: if target is one hot than it is assumed that the target is one-hot encoded
        include_classes: list of classes to include
        I2CVB_basedir: basepath above patients
    Returns:
        list of patient folder names
    """
    if target_one_hot:
        if torch.is_tensor(I2CVB_dataset[0][1]):
            weights = torch.zeros(len(include_classes))

            for p_idx in range(len(I2CVB_dataset)):
                img, seg, _ = I2CVB_dataset[p_idx]
                weights += torch.sum(seg, dim=(1, 2))
            weights /= 255
            weights = weights / torch.sum(weights)
        else:
            # numpy
            weights = np.zeros(len(include_classes))
            for p_idx in range(len(I2CVB_dataset)):
                img, seg, _ = I2CVB_dataset[p_idx]
                weights += np.sum(seg, axis=(1, 2))

            weights /= 255
            weights = weights / np.sum(weights)
    else:
        if torch.is_tensor(I2CVB_dataset[0][1]):
            t_list = []
            for i in range(len(I2CVB_dataset)):
                img, t, _ = I2CVB_dataset[i]
                t_list.append(t.flatten())
            _, counts = torch.unique(torch.cat(t_list, dim=0), sorted=True, return_counts=True)
            weights = counts / torch.sum(counts)
        else:
            # numpy
            t_list = []
            for i in range(len(I2CVB_dataset)):
                img, t, _ = I2CVB_dataset[i]
                t_list.append(t.flatten())
            _, counts = np.unique(np.concatenate(t_list, axis=0), return_counts=True)
            weights = counts / np.sum(counts)

    return weights


def plot_segmentation2_I2CVB(image, segmentation, include_classes=None,
                             show_ticks=True, title=None, save_name=None, save_only=False):
    r"""
    Plot a segmentation map on top of an image. And optionally save this image to disk. The channels of the segmentation
    map must be in the order of BG, LES, PZ, PRO
    Args:
        save_only: if save_only is set to True the image is only stored to disk
        show_ticks: Displays the ticks if set to True
        image: underlaying image tensor of shape H x W
        segmentation: overlaying one-hot encoded segmentation map tensor of shape C x H x W
        save_name: Directory where the image is saved to.

    Returns: None
    Examples:

    """
    if torch.is_tensor(image):
        image = image.detach().numpy()
    if torch.is_tensor(segmentation):
        segmentation = segmentation.detach().numpy()

    if include_classes is None:
        include_classes = ['bg', 'cap', 'pz', 'cg', 'prostate']

    rgb_dict = {
        'bg': [0, 0, 0],
        'cap': [255, 0, 0],
        'pz': [0, 255, 0],
        'cg': [0, 0, 255],
        'prostate': [255, 255, 0]
    }

    seg_map = np.argmax(segmentation, axis=0)

    rgb_values = [rgb_dict[include_classes[p]] for p in seg_map.flatten()]

    seg_map_rgb = np.array(rgb_values).reshape(seg_map.shape[0], seg_map.shape[1], 3)

    # Mask background 'BG'
    seg_map_masked = np.ma.masked_where(seg_map_rgb == rgb_dict['bg'], seg_map_rgb)

    # imsave(save_name, seg_map_masked)

    plt.figure()
    plt.title(title, fontsize=16)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.imshow(seg_map_masked, interpolation='none', alpha=0.3)

    if show_ticks is False:
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    # save figure
    if save_name is not None:
        plt.savefig(save_name)
    # plt.colorbar()

    if save_only:
        plt.close()
    else:
        plt.show()


def plot_segmentation_contour(image, segmentation, include_classes=None,
                              show_ticks=True, title=None, save_name=None, save_only=False):
    r"""
    Plot a segmentation contour on top of an image. And optionally save this image to disk. The channels of the segmentation
    map must be in the order of BG, LES, PZ, PRO
    Args:
        save_only: if save_only is set to True the image is only stored to disk
        show_ticks: Displays the ticks if set to True
        image: underlaying image tensor of shape H x W
        segmentation: overlaying one-hot encoded segmentation map tensor of shape C x H x W
        save_name: Directory where the image is saved to.

    Returns: None
    Examples:

    """
    import cv2

    if include_classes is None:
        include_classes = ['bg', 'cap', 'pz', 'cg', 'prostate']

    if torch.is_tensor(image):
        image = image.detach().numpy()
    if torch.is_tensor(segmentation):
        # https://scikit-image.org/docs/stable/user_guide/data_types.html#working-with-opencv
        segmentation = img_as_ubyte(segmentation.detach().numpy())

    # min max feature scaling image
    image = img_as_ubyte(feature_scaling(image))

    # convert image to rgb to show surroundings
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    rgb_dict = {
        'bg': [0, 0, 0],
        'cap': [255, 0, 0],
        'pz': [0, 255, 0],
        'cg': [0, 0, 255],
        'prostate': [255, 255, 0]
    }

    if len(segmentation) == len(include_classes):
        for i in range(len(include_classes)):
            contours, hierarchy = cv2.findContours(segmentation[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, rgb_dict[include_classes[i]], 1)

        plt.figure()
        plt.title(title, fontsize=16)
        plt.imshow(image, cmap='gray', interpolation='none')

        if show_ticks is False:
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        # save figure
        if save_name is not None:
            plt.savefig(save_name)
            # plt.colorbar()

        if save_only:
            plt.close()
        else:
            plt.show()

    else:
        print(f"segmentation len {segmentation.shape} differes from include classes len {len(include_classes)}.")


def plot_prediction_contour(image, groundtruth_seg, prediction_seg, include_classes=None,
                            show_ticks=True, title=None, save_name=None, save_only=False):
    r"""
    Plot a segmentation contour on top of an image. And optionally save this image to disk. The channels of the segmentation
    map must be in the order of BG, LES, PZ, CG, PRO
    Args:
        groundtruth_seg: groundtruth segmentation map one hot encoded
        save_only: if save_only is set to True the image is only stored to disk
        show_ticks: Displays the ticks if set to True
        image: underlaying image tensor of shape H x W
        prediction_seg: overlaying one-hot encoded segmentation map tensor of shape C x H x W
        save_name: Directory where the image is saved to.

    Returns: None
    Examples:

    """
    import cv2

    if include_classes is None:
        include_classes = ['bg', 'cap', 'pz', 'cg', 'prostate']

    if torch.is_tensor(image):
        image = image.detach().numpy()
    if torch.is_tensor(groundtruth_seg):
        groundtruth_seg = img_as_ubyte(groundtruth_seg.detach().numpy())
    if torch.is_tensor(prediction_seg):
        prediction_seg = img_as_ubyte(prediction_seg.detach().numpy())

    # min max feature scaling image
    image = img_as_ubyte(feature_scaling(image))

    # convert image to rgb to show surroundings
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    target_color = [0, 0, 255]
    pred_color = [0, 255, 0]

    if len(include_classes) < 4:
        f, axarr = plt.subplots(1, len(include_classes))
    else:
        cols = 3
        rows = len(include_classes) % cols;
        f, axarr = plt.subplots(rows, cols)
    if title is not None:
        f.suptitle(title, fontsize=16)

    if len(prediction_seg) == len(include_classes) and len(prediction_seg) == len(groundtruth_seg):
        if len(include_classes) < 4:
            for i in range(len(include_classes)):
                image_canvas = np.copy(image)
                # groundtruth contours
                contours, hierarchy = cv2.findContours(groundtruth_seg[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_canvas, contours, -1, target_color, 1)
                # prediction contours
                contours, hierarchy = cv2.findContours(prediction_seg[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_canvas, contours, -1, pred_color, 1)
                axarr[i].set_title(f"{include_classes[i]}", fontsize=16)
                axarr[i].imshow(image_canvas, cmap="gray")
        else:
            for i in range(len(include_classes)):
                image_canvas = np.copy(image)
                # groundtruth contours
                contours, hierarchy = cv2.findContours(groundtruth_seg[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_canvas, contours, -1, target_color, 1)
                # prediction contours
                contours, hierarchy = cv2.findContours(prediction_seg[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image_canvas, contours, -1, pred_color, 1)
                col = i % cols
                row = i // cols
                axarr[row, col].set_title(f"{include_classes[i]}", fontsize=16)
                axarr[row, col].imshow(image_canvas, cmap="gray")

        if show_ticks is False:
            if len(include_classes) < 4:
                for a in axarr:
                    a.axis('off')
            else:
                for a in axarr:
                    for ar in a:
                        ar.axis('off')

        plt.tight_layout()
        # save figure
        if save_name is not None:
            plt.savefig(save_name)
            # plt.colorbar()

        if save_only:
            plt.close()
        else:
            plt.show()

    else:
        assert False, (f"segmentation len {prediction_seg.shape} differes from include classes len {len(include_classes)}.")


def illustrate_weights():
    import pandas as pd
    import seaborn as sns
    target_one_hot = True
    num_of_surrouding_imgs = 1
    include_classes = ['cap', 'pz', 'cg', 'prostate']
    mic = ['cap', 'pz', 'cg', 'prostate']

    transform_train = Compose([
        ToTensor(),
        Resize((320, 320))
        # RandomHorizontalFlip(),
        # RandomCrop(10),
        # RandomRotation(80),
        # RandomGaussianNoise(0.0001)
    ])

    ds_name_keys = ['I2CVB', 'UDE', 'combined']
    weights = {}
    for ds_name_key in ds_name_keys:
        path = ds_path_dict['local']['clahe'][ds_name_key]
        ps = get_I2CVB_dataset(path)
        my_ds = DsI2CVB(I2CVB_basedir=path, include_patients=ps, mr_sequence='T2W', transform=transform_train,
                        include_classes=include_classes, num_of_surrouding_imgs=num_of_surrouding_imgs,
                        target_one_hot=target_one_hot, samples_must_include_classes=mic)

        weights[ds_name_key] = calc_i2cvb_weights(my_ds, mic, target_one_hot=target_one_hot).numpy()

    data = []
    for key in ds_name_keys:
        for class_i in range(len(mic)):
            data.append([key, mic[class_i], 100.0 * weights[key][class_i]])

    df = pd.DataFrame(data, columns=['dataset', 'class', 'percentage'], dtype=float)

    print(df)

    sns.set(font_scale=2.6)
    sns.set_style("whitegrid")
    g = sns.catplot(data=df, kind="bar", x='class', y='percentage', hue="dataset", palette="tab10", height=11.11, aspect=1.62)

    g.despine(left=True)
    g.set_axis_labels("class", "[%]")
    g.legend.set_title("dataset")
    g.ax.tick_params(axis='both', which='major')
    g.ax.axes.grid(b=True, axis='both')
    plt.show()


if __name__ == '__main__':
    pass
