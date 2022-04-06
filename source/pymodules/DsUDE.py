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
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset



class UDEDataset(Dataset):
    def __init__(self, ude_dataset_path, transform=None, num_of_surrouding_imgs=0,
                 include_classes=['BG', 'PRO', 'PZ', 'LES'], target_one_hot=True):

        self.ude_dataset_path = ude_dataset_path
        self.transform = transform
        self.num_of_surrouding_imgs = num_of_surrouding_imgs
        self.include_classes = include_classes
        self.target_one_hot = target_one_hot

        with open(self.ude_dataset_path, 'rb') as f:
            patient_dict = pickle.load(f)

        # data storage for all images and segmentations
        self.p_dict = {}

        # p_dict_index stores the indices of each patient sample
        # p_id: patient id, s_idx: sample index
        # create a mapping {0: {'p_id': 0, 's_idx': 0}, ..., 27: {'p_id': 2, 's': 34}, ...}
        self.p_dict_index = {}
        # create a mapping dict index is p_id, value is length of images in patient number p_id
        # is used for sequence generation to find corner cases
        self.p_dict_length = {}
        # patient_id in p_dict
        p_idx = 0

        # continous counter over all elements in p_dict used for p_dict_index
        sample_id_counter = 0
        for key in patient_dict:
            # key is index = 0, 1, 2, ..., N (patients)

            # save also lesions -> segmentation map has 4 dim
            # now create the one-hot encoded segmentation map and reduce to start and stop index, since
            # all the other images do not contain segmentations
            # one hot encode segmap to save original segmentation maps
            # seg_map shape: N X C x H x W
            seg_list = []
            for seg_class in patient_dict[key]:
                if seg_class in self.include_classes:
                    # we have to make sure that that the classes are appended in the following order: BG, LES, PZ, PRO
                    # the correct order is already given by the order in patient_dict[key]
                    seg_list.append(np.expand_dims(patient_dict[key][seg_class], axis=1))

            seg_map = np.concatenate(seg_list, axis=1)

            self.p_dict[p_idx] = {'id': patient_dict[key]['id'],
                                  'img': patient_dict[key]['img'],
                                  'seg': seg_map}

            # now fill up p_dict_index
            length_img_data = len(patient_dict[key]['img'])
            for sample_id in range(length_img_data):
                self.p_dict_index[sample_id_counter] = {'p_id': p_idx, 's_idx': sample_id}
                sample_id_counter += 1

            # now add new length to self.p_dict_length
            self.p_dict_length[p_idx] = length_img_data
            p_idx += 1

        del patient_dict

    def __len__(self):
        r"""
        Returns the length of the dataset
        """
        return len(self.p_dict_index)

    def __getitem__(self, idx):
        # it is assumed that each patient has at least two sample images
        p_idx, sample_id = self.p_dict_index[idx]['p_id'], self.p_dict_index[idx]['s_idx']
        p_max_samples = self.p_dict_length[p_idx]

        # sequence generation
        # find out at which index the current index is:
        current_img = self.p_dict[p_idx]['img'][sample_id]
        target = self.p_dict[p_idx]['seg'][sample_id]

        if self.num_of_surrouding_imgs == 0:
            sample_img = current_img
        elif self.num_of_surrouding_imgs == 1:
            # corner cases
            # sample_id == 0 or last element
            if sample_id == 0:
                # first image
                previous_img = np.copy(current_img)
                next_img = self.p_dict[p_idx]['img'][sample_id + 1]
            elif sample_id == p_max_samples - 1:
                # last image
                previous_img = self.p_dict[p_idx]['img'][sample_id - 1]
                next_img = np.copy(current_img)
            else:
                # image in b/W
                previous_img = self.p_dict[p_idx]['img'][sample_id - 1]
                next_img = self.p_dict[p_idx]['img'][sample_id + 1]

            # create array of shape H x W x C this is necessary for to TF.ToTensor()
            sample_img = np.concatenate((np.expand_dims(previous_img, axis=2),
                                         np.expand_dims(current_img, axis=2),
                                         np.expand_dims(next_img, axis=2)), axis=2)

        if self.transform:
            sample_img, target = self.transform(sample_img, target)
            # if transform is applied we have to recreate the background class since, e.g., the rotation creates zeros
            # at the area where the image is rotated out of the image frame
            target[0] = (torch.sum(target[1:], dim=0, dtype=target[0].dtype) == 0) * 1

        if not self.target_one_hot:
            if torch.is_tensor(target):
                target = torch.argmax(target, dim=0)
            else:
                target = np.argmax(target, axis=0)

        return sample_img, target


if __name__ == '__main__':
    pass
