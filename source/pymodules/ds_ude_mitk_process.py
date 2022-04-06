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
import glob
import pickle
import zipfile
from subprocess import Popen, PIPE
import numpy as np
import nrrd
import SimpleITK as sitk
from scipy.signal import medfilt2d
from skimage import img_as_ubyte, img_as_float32, img_as_uint
import cv2
from DsI2CVB import plot_segmentation2_I2CVB, plot_segmentation_contour
import random
import math


def ude_unzip_archives(ude_basedir):
    r"""
    unzip all UDE zip files in the folder ude_basedir
    Args:
        ude_basedir:

    Returns:

    """
    gl_pathname = os.path.join(ude_basedir, '*.zip')

    downloaded_archives = sorted([s for s in glob.glob(gl_pathname)])

    for i, s in enumerate(downloaded_archives):
        # extract files to
        try:
            with zipfile.ZipFile(s) as zip_ref:
                zip_ref.extractall(ude_basedir)
            print(f"#{i}: {os.path.basename(s)} processed")
        except AssertionError:
            print(f"dir {os.path.basename(s)} already exists - skipped")


def mitk_process(ude_basedir, dataset_name, mitk_file_converter_filepath, include_only_class_list=None,
                 exclude_shapes=None):
    r"""
    This function extracts all the mitk files into one single pickle file that contains all the mr sequences
    with all segmentation maps and sample images
    Args:
        dataset_name: name of the dataset
        exclude_shapes: is a dict that contains information which shape (H x W) to exclude for which MR sequence
        include_only_class_list: if this list is set to one or more  class(es) then only the images and segmentation
        maps are added for which at least one class is present.
        ude_basedir: Directory of the Dataset
        mitk_file_converter_filepath: Filepath to the MITK File Converter utility

    Returns:

    """
    possible_include_classes = ['BG', 'LES', 'PZ', 'PRO']

    extracted_archives = sorted([f.path for f in os.scandir(ude_basedir) if f.is_dir()])

    # initialize the dictonary that will contain all images and segmentation data
    patient_dict = {}

    # p_idx is the index for the dict that is exported it contains only valid images according to
    # shape, segmentations and ...
    p_idx = 0

    for i, f in enumerate(extracted_archives, 0):
        gl_pathname = os.path.join(f, '*.mitk')
        mitk_archives = sorted([os.path.basename(s) for s in glob.glob(gl_pathname)])

        print(f"#{i}: {os.path.basename(f)}:")

        # first find all MRI types for the current directory (current patient)
        # MRI types are encoded in the first index of the *.mitk name
        # examples are: ADC, T2, DCE
        mri_types = set([l.split('.mitk')[0].split('_')[1] for l in mitk_archives])
        mri_dict = dict()
        for mt in mri_types:
            mri_dict[mt] = 0

        # mri_dict = {'ADC': 0, 'DCE': 0}
        # now we have a dict that contains all the MRI types present in the current folder
        # this is necessary to know since each mitk contains a two sets of *.nrrds
        # input most likely images: *.nrrd most likely segmentation maps: *_1.nrrd
        # one *.nrrd is the input images the other one is the segmentation map
        # as soon as one *.nrrd pair is processed for a given MRI type the value in mri_dict
        # is set to 1, then we know that we do not have to read in any further input images
        # for a given MRI type

        # create an empty dict that stores all the images and segmentations for the current patient
        # # patient_dict = {'id':  2.25.169499415547834839663877919141036099853,
        #                 'T2': {'img': np.ndarray(), 'Pro': np.ndarray(), 'Pz': ...}
        #                  'ADC': {...}, ...}
        current_patient_dict = {'patient_id': os.path.basename(f)}

        for l in mitk_archives:
            # mitk_archives = ['seg_ADC_PRO.mitk', 'seg_ADC_PZ.mitk', 'seg_DCE_PRO.mitk', 'seg_DCE_PZ.mitk']
            #                        Segmentation, Type (ADC, DCE, T2), Class
            # e.g. seg_ADC_LES1.mitk -> seg_ADC_LES1
            a = l.split('.mitk')[0]
            # -> ['seg', 'ADC', 'LES1']
            a_type = a.split('_')

            # Extract nrrd files using MitkFileConverter.sh
            input_mitk = os.path.join(f, l)
            output_nrrd = os.path.join(f, f"{a}.nrrd")
            process = Popen([mitk_file_converter_filepath, '-i', input_mitk, '-o', output_nrrd],
                            stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            if process.wait() == 1:
                # Error occured
                print(stderr)
            else:
                # Process *.nrrd files
                output_1_nrrd = os.path.join(f, f"{a}_1.nrrd")

                if os.path.isfile(output_nrrd) and os.path.isfile(output_1_nrrd):
                    # 1. check if two *.nrrd files were created
                    # now go on w/ processing the *.nrrd files
                    # check if it is necessary to read in the image files
                    data_output, header_output = nrrd.read(output_nrrd, index_order='C')
                    data_output_1, header_output_1 = nrrd.read(output_1_nrrd, index_order='C')

                    # Two *nrrd files were created. Most likely the *.nrrd file contains the
                    # images and the segmentation nrrd is most likely the *_1.nrrd file but we are not sure
                    # we have to check in the header the 'modality' key to make sure
                    # image data: header_data['modality']: 'MR'
                    # label data: header_data['modality']: 'org.mitk.image.multilabel'
                    if header_output['modality'] == 'MR' and header_output_1['modality'] == 'org.mitk.image.multilabel':
                        # create new subdict for MRI type
                        data_img = data_output
                        data_label = data_output_1
                    elif header_output['modality'] == 'org.mitk.image.multilabel' and header_output_1[
                        'modality'] == 'MR':
                        data_img = data_output_1
                        data_label = data_output
                    else:
                        print("Error in dataset generation: Could not find a valid dataset.")
                        break

                    if mri_dict[a_type[1]] == 0:
                        # read images if not yet processed
                        # create new subdict for MRI type
                        # https://scikit-image.org/docs/dev/user_guide/data_types.html
                        # convert all images to float
                        # current_patient_dict[a_type[1]] = {'img': np.array(img_as_float32(data_img))}
                        current_patient_dict[a_type[1]] = {'img': np.array(img_as_uint(data_img))}
                        # set mri_dict["type"] to one meaning that the image files were already copied
                        mri_dict[a_type[1]] = 1

                    # add segmentation to dict
                    # segmentation maps are uint8
                    current_patient_dict[a_type[1]][a_type[2]] = np.array(img_as_ubyte(data_label))
                    # now clean up delete both *.nrrd files
                    os.remove(output_nrrd)
                    os.remove(output_1_nrrd)
                else:
                    print(f"error for {l}: could not create two *.nrrd files.")

                print(f"\t{l} processed.")

        # now the current_patient dict is filled and looks like
        # {'patient_id': '2.25.106158470415091923704066038061253004463',
        # 'ADC': {'img': array(N x H x W), dtype=uint16),
        #     'PRO': array(N x H x W), dtype=uint16),
        #     'PZ': array(N x H x W, dtype=uint16)},
        # 'DCE': {'img': array(3 x N x H x W, dtype=uint16),
        #     'PRO': array(N x H x W, dtype=uint16),
        #     'PZ': array(N x H x W, dtype=uint16)}}
        # now we sort the dict and create segmentation maps from it
        # print(current_patient_dict)

        # sort order for the temporaryly created dict p_dict in order to create a
        # background class and the lesion class which is composed by all lesions LES1, LES2, LES3
        sort_order = ['img', 'LES', 'PZ', 'PRO']
        # create background class from bg_seg list classes
        bg_seg = ['LES', 'PZ', 'PRO']
        # switch to track whether data was inserted to the patient_dict or not
        have_inserted_data = False

        possible_mr_sequences = ['ADC', 'DCE', 'T2']

        for mr_sequence in current_patient_dict:
            if mr_sequence == 'ADC' or mr_sequence == 'T2':
                # ADC or T2 are of shape N x H x W
                print('\t processing sequence: ', mr_sequence)
                # only use the samples with consistent shape. Otherwise they cannot be used by the model,
                # some samples have mixed shapes such as 768 768 22 and 640 640 42 (id =
                # 2.25.143647086371506238144966069408882640104) therefore all segmentations are compared against the
                # sample images
                # switch_use_shape switch to check if dimensionality fits and shapes of seg maps match
                switch_use_shape = True
                img_shape = current_patient_dict[mr_sequence]['img'].shape
                print('img_shape', img_shape)

                if len(img_shape) != 3:
                    print(f"image has wrong dimensionalitiy. must have 3 (but has {len(img_shape)}).")
                    switch_use_shape = False
                else:
                    # len == 3
                    for seg_type in current_patient_dict[mr_sequence]:
                        if seg_type != 'img':
                            # compare only with segmentation maps
                            if current_patient_dict[mr_sequence][seg_type].shape != img_shape:
                                switch_use_shape = False
                                break

                if exclude_shapes is not None:
                    if mr_sequence in exclude_shapes:
                        if img_shape[1:] in exclude_shapes[mr_sequence]:
                            switch_use_shape = False

                if switch_use_shape:
                    # the image and the segmentation have the same shape
                    # print(patient_id)
                    # we create a new dict that does only contain the mr sequence type specified in the
                    # constructor w/ mr_sequence and already a segmentation map with channels
                    # channel 0: BG, channel 1: les, channel 2: PZ, channel 3: PRO

                    # p_dict_sorted only contains keys of sort_order with background class
                    # bg generated from ['LES', 'PZ', 'PRO']
                    p_dict_sorted = {'img': [], 'BG': [], 'LES': [], 'PZ': [], 'PRO': []}

                    p_dict = current_patient_dict[mr_sequence]

                    dict_keys = sorted([key for key in p_dict])

                    # order the segmentation map according to sort_order
                    # overlay all LESx on top of each other
                    for key_so in sort_order:
                        for k in dict_keys:
                            if k.startswith(key_so):
                                # here we add images and segmentations maps to lists, in case there are
                                # three lesions LES1, LES2, LES3 they will be concatenated along the
                                # channel dimension in order to obtain one single lesion class. Here
                                # we assume that the lesion texture is the same for all lesions LES1,
                                # LES2, LES3
                                p_dict_sorted[key_so].append(p_dict[k])

                    for key in p_dict_sorted:
                        if len(p_dict_sorted[key]) == 0:
                            # list is empty fill segmentation with zeros
                            p_dict_sorted[key] = np.zeros(img_shape, dtype=np.uint8)
                        elif len(p_dict_sorted[key]) > 1:
                            # more than one element in list, sum across the channel axis to
                            # create one lesion segmentation map
                            p_dict_sorted[key] = np.sum(p_dict_sorted[key], axis=0, dtype=np.uint8).squeeze()
                        else:
                            # only one element in list
                            p_dict_sorted[key] = p_dict_sorted[key][0]

                    # create background class from bg_seg list classes
                    bg_list = [p_dict_sorted[c] for c in bg_seg]
                    p_dict_sorted['BG'] = (np.sum(bg_list, axis=0, dtype=np.uint8).squeeze() == 0) * 1

                    # clean up all the images that do not contain segmentation maps. We look for the first and
                    # last image we make use of the background since the background does contain all segmentations on
                    # top of each other
                    have_inserted_data = True

                    if include_only_class_list is not None:
                        # check if include_only_class_list is a subset of possible_include_classes
                        if all(cls_x in possible_include_classes for cls_x in include_only_class_list):
                            # only include images for which the segmentation include_only_class is present.
                            include_indices = []

                            for img_idx in range(len(p_dict_sorted['BG'])):
                                # Background is present in all images therefore we can use it to iterate over all
                                # segmentations
                                or_gate_classes = 0
                                for cls_x in include_only_class_list:
                                    or_gate_classes += np.sum(p_dict_sorted[cls_x][img_idx])
                                if or_gate_classes > 0:
                                    include_indices.append(img_idx)
                            print('\t \t', 'p_idx: ', p_idx, 'include_indices: ', len(include_indices), include_indices)

                            # create an empty dict which will contain later the complete segmentation maps for each
                            # MRI scan type with all segmentations
                            # e.g.
                            # patient_dict = {'id':  2.25.169499415547834839663877919141036099853,
                            #                'T2': {'img': np.ndarray(), 'BG': np.array..., 'LES', 'PZ', Pro': np.ndarray()}
                            #                'ADC': {...}, ...}
                            if have_inserted_data:
                                if p_idx not in patient_dict:
                                    # create key first if not yet created
                                    patient_dict[p_idx] = {}
                                if mr_sequence not in patient_dict[p_idx]:
                                    patient_dict[p_idx][mr_sequence] = {}
                                # insert segmentation data
                                for key in p_dict_sorted:
                                    if key == 'img':
                                        # update p_dict_sorted images that have a segmentation in LES, PZ, PRO
                                        if len(p_dict_sorted['img'].shape) == 3:
                                            # ADC and T2 are of shape N x H x W
                                            patient_dict[p_idx][mr_sequence][key] = p_dict_sorted['img'][
                                                include_indices]
                                        else:
                                            print("error with image shape!")
                                            raise AssertionError
                                    else:
                                        # key is a segmentation update the "raw" segmentations this means that a pixel does not
                                        # have a unique class but sometimes more than one class. This is fixed in UDEDataset
                                        patient_dict[p_idx][mr_sequence][key] = p_dict_sorted[key][include_indices]
                    else:
                        # include all images and segmentation maps
                        if have_inserted_data:
                            if p_idx not in patient_dict:
                                # create key first if not yet created
                                patient_dict[p_idx] = {}
                            if mr_sequence not in patient_dict[p_idx]:
                                patient_dict[p_idx][mr_sequence] = {}
                            # insert segmentation data
                            for key in p_dict_sorted:
                                if key == 'img':
                                    # update p_dict_sorted images that have a segmentation in LES, PZ, PRO
                                    if len(p_dict_sorted['img'].shape) == 3:
                                        # ADC and T2 are of shape N x H x W
                                        patient_dict[p_idx][mr_sequence][key] = p_dict_sorted['img']
                                    else:
                                        print("error with image shape!")
                                        raise AssertionError
                                else:
                                    # key is a segmentation update the "raw" segmentations this means that a pixel
                                    # does not have a unique class but sometimes more than one class. This is fixed
                                    # in UDEDataset
                                    patient_dict[p_idx][mr_sequence][key] = p_dict_sorted[key]
                else:
                    print(
                        f"{i} has inconsistent shape or is excluded (segtype: {seg_type} {current_patient_dict[mr_sequence][seg_type].shape} vs. img {img_shape}).")
                    # pass
            elif mr_sequence == 'DCE':
                # DCE is of shape H x W x N x C
                print('\t processing sequence: ', mr_sequence)
                # only use the samples with consistent shape. Otherwise they cannot be used by the model,
                # some samples have mixed shapes such as 768 768 22 and 640 640 42 (id =
                # 2.25.143647086371506238144966069408882640104) therefore all segmentations are compared against the
                # sample images
                switch_use_shape = True
                img_shape = current_patient_dict[mr_sequence]['img'].shape
                print('img_shape', img_shape)

                seg_shape = img_shape[1:]
                if len(img_shape) != 4:
                    print(f"image has wrong dimensionalitiy. must have 4 (but has {len(img_shape)}).")
                    switch_use_shape = False
                else:
                    for seg_type in current_patient_dict[mr_sequence]:
                        if seg_type != 'img':
                            # compare only with segmentation maps
                            if current_patient_dict[mr_sequence][seg_type].shape != seg_shape:
                                switch_use_shape = False
                                break

                if exclude_shapes is not None:
                    if mr_sequence in exclude_shapes:
                        if img_shape[2:] in exclude_shapes[mr_sequence]:
                            switch_use_shape = False

                if switch_use_shape:
                    # the image and the segmentation have the same shape
                    # print(patient_id)
                    # we create a new dict that does only contain the mr sequence type specified in the
                    # constructor w/ mr_sequence and already a segmentation map with channels
                    # channel 0: BG, channel 1: les, channel 2: PZ, channel 3: PRO

                    # p_dict_sorted only contains keys of sort_order with background class
                    # bg generated from ['LES', 'PZ', 'PRO']
                    p_dict_sorted = {'img': [], 'BG': [], 'LES': [], 'PZ': [], 'PRO': []}

                    p_dict = current_patient_dict[mr_sequence]

                    dict_keys = sorted([key for key in p_dict])

                    # order the segmentation map according to sort_order
                    # overlay all LESx on top of each other
                    for key_so in sort_order:
                        for k in dict_keys:
                            if k.startswith(key_so):
                                # here we add images and segmentations maps to lists, in case there are
                                # three lesions LES1, LES2, LES3 they will be concatenated along the
                                # channel dimension in order to obtain one single lesion class. Here
                                # we assume that the lesion texture is the same for all lesions LES1,
                                # LES2, LES3
                                p_dict_sorted[key_so].append(p_dict[k])

                    for key in p_dict_sorted:
                        if len(p_dict_sorted[key]) == 0:
                            # list is empty fill segmentation with zeros
                            p_dict_sorted[key] = np.zeros(seg_shape, dtype=np.uint8)
                        elif len(p_dict_sorted[key]) > 1:
                            # more than one element in list, sum accross the channel axis to
                            # create one lesion segmentation map
                            p_dict_sorted[key] = np.sum(p_dict_sorted[key], axis=0, dtype=np.uint8).squeeze()
                        else:
                            # only one element in list
                            p_dict_sorted[key] = p_dict_sorted[key][0]

                    # create background class from bg_seg list classes
                    bg_list = [p_dict_sorted[c] for c in bg_seg]
                    p_dict_sorted['BG'] = (np.sum(bg_list, axis=0, dtype=np.uint8).squeeze() == 0) * 1

                    # clean up all the images that do not contain segmentation maps. We look for the first and
                    # last image we make use of the background since the background does contain all segmentations on
                    # top of each other
                    have_inserted_data = True

                    if include_only_class_list is not None:
                        # check if include_only_class_list is a subset of possible_include_classes
                        if all(cls_x in possible_include_classes for cls_x in include_only_class_list):
                            # only include images for which the segmentation include_only_class is present.
                            include_indices = []

                            for img_idx in range(len(p_dict_sorted['BG'])):
                                # Background is present in all images therefore we can use it to iterate over all
                                # segmentations
                                or_gate_classes = 0
                                for cls_x in include_only_class_list:
                                    or_gate_classes += np.sum(p_dict_sorted[cls_x][img_idx])
                                if or_gate_classes > 0:
                                    include_indices.append(img_idx)
                            print('\t \t', 'p_idx: ', p_idx, 'include_indices: ', len(include_indices), include_indices)

                            # create an empty dict which will contain later the complete segmentation maps for each
                            # MRI scan type with all segmentations
                            # e.g.
                            # patient_dict = {'id':  2.25.169499415547834839663877919141036099853,
                            #                'T2': {'img': np.ndarray(), 'BG': np.array..., 'LES', 'PZ', Pro': np.ndarray()}
                            #                'ADC': {...}, ...}
                            if have_inserted_data:
                                if p_idx not in patient_dict:
                                    # create key first if not yet created
                                    patient_dict[p_idx] = {}
                                if mr_sequence not in patient_dict[p_idx]:
                                    patient_dict[p_idx][mr_sequence] = {}
                                # insert segmentation data
                                for key in p_dict_sorted:
                                    if key == 'img':
                                        # update p_dict_sorted images that have a segmentation in LES, PZ, PRO
                                        if len(p_dict_sorted['img'].shape) == 4:
                                            # DCE is of shape 3 x N x H x W -> slice out only N dimension
                                            patient_dict[p_idx][mr_sequence][key] = p_dict_sorted['img'][:,
                                                                                    include_indices]
                                        else:
                                            print("error with image shape!")
                                            raise AssertionError
                                    else:
                                        # key is a segmentation update the "raw" segmentations this means that a
                                        # pixel does not have a unique class but sometimes more than one class. This
                                        # is fixed in UDEDataset
                                        patient_dict[p_idx][mr_sequence][key] = p_dict_sorted[key][include_indices]
                    else:
                        # include all images and segmentation maps
                        if have_inserted_data:
                            if p_idx not in patient_dict:
                                # create key first if not yet created
                                patient_dict[p_idx] = {}
                            if mr_sequence not in patient_dict[p_idx]:
                                patient_dict[p_idx][mr_sequence] = {}
                            # insert segmentation data
                            for key in p_dict_sorted:
                                if key == 'img':
                                    # update p_dict_sorted images that have a segmentation in LES, PZ, PRO
                                    if len(p_dict_sorted['img'].shape) == 4:
                                        # DCE is of shape 3 x N x H x W -> slice out only N dimension
                                        patient_dict[p_idx][mr_sequence][key] = p_dict_sorted['img']
                                    else:
                                        print("error with image shape!")
                                        raise AssertionError
                                else:
                                    # key is a segmentation update the "raw" segmentations this means that a pixel
                                    # does not have a unique class but sometimes more than one class. This is fixed
                                    # in UDEDataset
                                    patient_dict[p_idx][mr_sequence][key] = p_dict_sorted[key]
                else:
                    print(
                        f"{i} has inconsistent shape or is excluded (segtype: {seg_type} {current_patient_dict[mr_sequence][seg_type].shape} vs. img {img_shape}).")

        if have_inserted_data:
            # data was inserted now add id and increment p_idx
            patient_dict[p_idx]['id'] = current_patient_dict['patient_id']
            p_idx += 1

        # stop after i patients
        # if i == 2:
        #  break

    # store patient_dict in a pickle file
    ude_dataset_filepath = os.path.join(ude_basedir, dataset_name)

    with open(ude_dataset_filepath, 'wb') as f:
        # Pickle the 'patient_dict' dictionary using the highest protocol available.
        pickle.dump(patient_dict, f, pickle.HIGHEST_PROTOCOL)

    del patient_dict


def train_test_split(raw_pickle_filepath, ude_basedir, mr_sequence='T2', train_rel_size=0.75,
                     exclude_patient_ids: list = None):
    r"""
    always use one patient in either train or test set

    choose mr sequence
    choose split for 80 / 20 -> at least 80 % of the images are in the testset
    create two files: ude_dataset_t2_train.pickle / ude_dataset_t2_test.pickle

    Args:
        exclude_patient_ids: list of patient id strings that are excluded from the final dataset
        ude_basedir: export directory
        raw_pickle_filepath: filepath to the raw pickle file that was created by mitk_process
        mr_sequence: The sequence that is to be extracted
        train_rel_size: minimum relative size of the training data

    Returns:

    """
    # initialise random seed
    random.seed(30)

    dataset_name = os.path.basename(raw_pickle_filepath).split('.pickle')[0]
    trainset_name = os.path.join(ude_basedir, f"{dataset_name}_train_{mr_sequence}.pickle")
    testset_name = os.path.join(ude_basedir, f"{dataset_name}_test_{mr_sequence}.pickle")

    print(f"Train- testsplit for {mr_sequence}:")
    with open(raw_pickle_filepath, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        patient_dict = pickle.load(f)

    # copy all the MR images that belong to the sequence mr_sequence
    patient_dict_sequence = {}

    p_idx = 0
    for key in patient_dict:
        # key is index = 0, 1, 2, ..., N (patients)
        if mr_sequence in patient_dict[key]:
            patient_id = patient_dict[key]['id']
            if exclude_patient_ids is not None:
                if patient_id not in exclude_patient_ids:
                    # mr_sequence is DCE, ADC, T2
                    patient_dict_sequence[p_idx] = patient_dict[key][mr_sequence]
                    patient_dict_sequence[p_idx]['id'] = patient_id
                    p_idx += 1
            else:
                # mr_sequence is DCE, ADC, T2
                patient_dict_sequence[p_idx] = patient_dict[key][mr_sequence]
                patient_dict_sequence[p_idx]['id'] = patient_id
                p_idx += 1

    # copy finished free memory
    del patient_dict

    # determine the patient id where to split the dataset
    count_dict = {}
    for key in patient_dict_sequence:
        s = patient_dict_sequence[key]['img'].shape
        if mr_sequence == 'DCE':
            count_dict[key] = s[1]
        else:
            count_dict[key] = s[0]

    total_imgs = 0
    total_keys = 0

    for key, value in count_dict.items():
        total_imgs += value
        total_keys += 1

    print('Found: total_keys:', total_keys, 'total_imgs:', total_imgs)

    # the rest will be added to the test set
    patient_dict_train = {}
    patient_dict_test = {}

    # e.g. total size: 885 train_size_in = 663
    train_size_min = math.floor(total_imgs * train_rel_size)

    # now we add randomly a patient until we have at least 663 samples in the train set
    train_size_current = 0

    # create a list of 29 random total_keys in b/w 0 and 29
    random_order = random.sample(range(0, total_keys), total_keys)

    # fill train dict
    for idx_new, idx_old in enumerate(random_order):
        patient_dict_train[idx_new] = patient_dict_sequence[idx_old]
        if mr_sequence == 'DCE':
            # DCE has shape 3 x N x H x W
            train_size_current += patient_dict_sequence[idx_old]['img'].shape[1]
        elif mr_sequence == 'ADC' or mr_sequence == 'T2':
            # ADC, T2 have shape N x H x W
            train_size_current += patient_dict_sequence[idx_old]['img'].shape[0]
        else:
            print(f'unknown sequence {mr_sequence}')

        if train_size_current >= train_size_min:
            # train dict is full
            break

    print('train set contains: ', train_size_current, f'images ({(100 * train_size_current / total_imgs):.2f} %). and '
                                                      f'ids:', random_order[:idx_new + 1])

    # fill test dict
    test_size_current = 0
    for idx_new2, idx_old2 in enumerate(random_order[idx_new + 1:]):
        patient_dict_test[idx_new2] = patient_dict_sequence[idx_old2]
        if mr_sequence == 'DCE':
            # DCE has shape 3 x N x H x W
            test_size_current += patient_dict_sequence[idx_old2]['img'].shape[1]
        else:
            # ADC, T2 have shape N x H x W
            test_size_current += patient_dict_sequence[idx_old2]['img'].shape[0]

    print('test set contains:', test_size_current, f'images ({(100 * test_size_current / total_imgs):.2f} %) ids:',
          random_order[idx_new + 1:])

    # store train / test patient dict in a pickle file
    pickle_filepath_train = os.path.join(ude_basedir, trainset_name)

    with open(pickle_filepath_train, 'wb') as f:
        # Pickle the 'patient_dict' dictionary using the highest protocol available.
        pickle.dump(patient_dict_train, f, pickle.HIGHEST_PROTOCOL)

    del patient_dict_train

    pickle_filepath_test = os.path.join(ude_basedir, testset_name)

    with open(pickle_filepath_test, 'wb') as f:
        # Pickle the 'patient_dict' dictionary using the highest protocol available.
        pickle.dump(patient_dict_test, f, pickle.HIGHEST_PROTOCOL)

    del patient_dict_test


def save_all_images(raw_pickle_filepath, mr_sequence, save_dir):
    include_classes = ['BG', 'LES', 'PZ', 'PRO']
    with open(raw_pickle_filepath, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        patient_dict = pickle.load(f)

    for patient_id in patient_dict:
        # patient_id is 0, 1, ..., N
        print(f"processing patient_id: {patient_id}")
        if mr_sequence in patient_dict[patient_id]:
            for img_idx in range(patient_dict[patient_id][mr_sequence]['img'].shape[0]):
                # img_idx is the index of the current image in the sequence of the patient patient_id
                img = patient_dict[patient_id][mr_sequence]['img'][img_idx]

                # save also lesions -> segmentation map has 4 dim
                # now create the label encoded segmentation map and reduce to start and stop index, since
                # all the other images do not contain segmentations
                # one hot encode segmap to save original segmentation maps
                seg_map = np.concatenate((np.expand_dims(patient_dict[patient_id][mr_sequence]['BG'][img_idx], axis=0),
                                          np.expand_dims(patient_dict[patient_id][mr_sequence]['LES'][img_idx], axis=0),
                                          np.expand_dims(patient_dict[patient_id][mr_sequence]['PZ'][img_idx], axis=0),
                                          np.expand_dims(patient_dict[patient_id][mr_sequence]['PRO'][img_idx],
                                                         axis=0)),
                                         axis=0)
                # save each image to disk
                save_name = os.path.join(save_dir, f"{patient_id}_{img_idx}.png")

                plot_segmentation_contour(image=img, segmentation=seg_map, include_classes=include_classes,
                                          show_ticks=False, title=f"{patient_id}_{img_idx}", save_name=save_name,
                                          save_only=True)


def ude_to_dicom(raw_pickle_filepath, export_basedir, mr_sequence='T2',
                 exclude_patient_ids=None,
                 create_preview=False):
    if exclude_patient_ids is None:
        exclude_patient_ids = ['2.25.143647086371506238144966069408882640104']

    include_classes = ['BG', 'LES', 'PZ', 'PRO']

    with open(raw_pickle_filepath, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        patient_dict = pickle.load(f)

    for patient_id in patient_dict:
        # patient_id is 0, 1, ..., N
        print(f"processing patient_id: {patient_id}")
        patient_uuid = patient_dict[patient_id]['id']
        if exclude_patient_ids is not None:
            if patient_uuid not in exclude_patient_ids:
                if mr_sequence in patient_dict[patient_id]:
                    # create patient directory
                    current_exportdir = os.path.join(export_basedir, patient_uuid)
                    current_exportdir_img = os.path.join(current_exportdir, 'T2W')
                    current_exportdir_seg = os.path.join(current_exportdir, 'GT')
                    current_exportdir_seg_cap = os.path.join(current_exportdir_seg, 'cap')
                    current_exportdir_seg_pro = os.path.join(current_exportdir_seg, 'prostate')
                    current_exportdir_seg_pz = os.path.join(current_exportdir_seg, 'pz')
                    current_exportdir_seg_cg = os.path.join(current_exportdir_seg, 'cg')

                    dir_list = [current_exportdir, current_exportdir_img, current_exportdir_seg,
                                current_exportdir_seg_cap,
                                current_exportdir_seg_pro, current_exportdir_seg_pz, current_exportdir_seg_cg]

                    print(f"export dir: {current_exportdir}")

                    for path in dir_list:
                        try:
                            os.mkdir(path)
                        except OSError as error:
                            print(error)

                    for img_idx in range(patient_dict[patient_id][mr_sequence]['img'].shape[0]):
                        # img_idx is the index of the current image in the sequence of the patient patient_id

                        # write image
                        # castFilter = sitk.CastImageFilter()
                        # castFilter.SetOutputPixelType(sitk.sitkInt16)
                        # Convert floating type image (imgSmooth) to int type (imgFiltered)
                        img_buf = patient_dict[patient_id][mr_sequence]['img'][img_idx]
                        img = sitk.GetImageFromArray(img_buf)
                        # img = castFilter.Execute(img)
                        sitk.WriteImage(img, os.path.join(current_exportdir_img, f"Image{img_idx}.dcm"))

                        # write segmentations
                        # cap / les
                        img = sitk.GetImageFromArray(patient_dict[patient_id][mr_sequence]['LES'][img_idx] * 255)
                        sitk.WriteImage(img, os.path.join(current_exportdir_seg_cap, f"Image{img_idx}.dcm"))
                        # pz / PZ
                        img_pz = sitk.GetImageFromArray(patient_dict[patient_id][mr_sequence]['PZ'][img_idx] * 255)
                        sitk.WriteImage(img_pz, os.path.join(current_exportdir_seg_pz, f"Image{img_idx}.dcm"))
                        # prostate / PRO
                        img_pro = sitk.GetImageFromArray(patient_dict[patient_id][mr_sequence]['PRO'][img_idx] * 255)
                        sitk.WriteImage(img_pro, os.path.join(current_exportdir_seg_pro, f"Image{img_idx}.dcm"))
                        # create the central gland class here
                        wg_map = 255 * patient_dict[patient_id][mr_sequence]['PRO'][img_idx]
                        pz_map = 255 * patient_dict[patient_id][mr_sequence]['PZ'][img_idx]
                        cg_map = wg_map - pz_map

                        # median filter cg_map
                        # kernel_size = 15 works well
                        # mean filter

                        # median filter cg_map
                        # kernel_size = 15 works well
                        # mean filter
                        alpha = 0.8
                        kernel_size = int(alpha * (np.sqrt(np.sum(cg_map / np.max(cg_map)) / np.pi)))

                        if kernel_size % 2 == 0:
                            kernel_size -= 1

                        print(f"img_idx: {img_idx}, kernel_size: {kernel_size}")

                        cg_map = medfilt2d(cg_map, kernel_size=kernel_size)

                        cg_map = np.array(cg_map - ((cg_map / 255) * (pz_map / 255)) * 255, dtype=np.uint8)

                        # find all contours and select the max contour
                        # sometimes the pc contains more than two contours extract the bigger one
                        pz_cnt, pz_hierarchy = cv2.findContours(pz_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # we have a pz contour otherwise cg_map is cg_map
                        if len(pz_cnt) > 0:
                            pz_cnt_max_idx = 0
                            pz_cnt_s = pz_cnt[0].shape[0]
                            for idx, pc in enumerate(pz_cnt):
                                if pc.shape[0] > pz_cnt_s:
                                    pz_cnt_s = pc.shape[0]
                                    pz_cnt_max_idx = idx

                            M = cv2.moments(pz_cnt[pz_cnt_max_idx])

                            pz_cx = int(M['m10'] / M['m00'])
                            pz_cy = int(M['m01'] / M['m00'])

                            # find all contours and select the max contour
                            contours, hierarchy = cv2.findContours(cg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            # select the contour with topmost / smallest y coordinate
                            contours_centre_distance = np.zeros(len(contours))
                            contours_area = np.zeros_like(contours_centre_distance)
                            for cnt_idx, cnt in enumerate(contours):
                                # contour moments
                                M = cv2.moments(cnt)
                                if M['m00'] == 0:
                                    contours_area[cnt_idx] = M['m00']
                                    contours_centre_distance[cnt_idx] = np.sqrt(pz_cx ** 2 + pz_cy ** 2)
                                else:
                                    # contour coordinates
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])

                                    contours_centre_distance[cnt_idx] = np.sqrt((pz_cx - cx) ** 2 + (pz_cy - cy) ** 2)
                                    contours_area[cnt_idx] = M['m00']

                            contour_max_area_idx = np.argmax(contours_area)
                            # select contour with smallest distance b/w pz centre and cg centres
                            contour_top_idx = np.argmin(contours_centre_distance)

                            if contours_area[contour_top_idx] >= 0.4 * contours_area[contour_max_area_idx]:
                                contour_top = contours[contour_top_idx]
                            else:
                                contour_top = contours[contour_max_area_idx]

                            # contour_top = contours[contour_top_idx]

                            cg_map = np.zeros_like(cg_map)
                            cv2.drawContours(cg_map, [contour_top], -1, color=(255, 255, 255), thickness=cv2.FILLED)

                        # or wavelet filter: the current wavelet configuration is set to an image size of 320 x 320
                        # if the sizes mismatch the wavelets have to be adapted
                        # cg_map = img_as_ubyte(wavelet_filtering(cg_map))
                        # add noise reduction here
                        # max value of cg_map is already 255
                        img_cg = sitk.GetImageFromArray(cg_map)
                        sitk.WriteImage(img_cg, os.path.join(current_exportdir_seg_cg, f"Image{img_idx}.dcm"))

                        if create_preview:
                            # create an image to check how well the cg creation worked
                            seg_map = np.concatenate(
                                (np.expand_dims(patient_dict[patient_id][mr_sequence]['BG'][img_idx], axis=0),
                                 np.expand_dims(patient_dict[patient_id][mr_sequence]['LES'][img_idx], axis=0),
                                 np.expand_dims(patient_dict[patient_id][mr_sequence]['PZ'][img_idx], axis=0),
                                 np.expand_dims(cg_map / 255, axis=0),
                                 np.expand_dims(patient_dict[patient_id][mr_sequence]['PRO'][img_idx],
                                                axis=0)),
                                axis=0)

                            # save each image to disk
                            save_name = os.path.join(export_basedir, f"{patient_id}_{img_idx}.png")

                            plot_segmentation2_I2CVB(image=img_buf, segmentation=seg_map,
                                                     include_classes=['bg', 'cap', 'pz', 'cg', 'prostate'],
                                                     show_ticks=False, title=f"{patient_uuid}_{img_idx}",
                                                     save_name=save_name,
                                                     save_only=True)


def ude_cg_extract(raw_pickle_filepath, export_basedir, mr_sequence='T2',
                   exclude_patient_ids=None,
                   create_preview=False):
    if exclude_patient_ids is None:
        exclude_patient_ids = ['2.25.143647086371506238144966069408882640104']

    include_classes = ['BG', 'LES', 'PZ', 'PRO']
    fileext = 'png'

    seg_dir1 = os.path.join(export_basedir, "seg_dir1")
    seg_dir2 = os.path.join(export_basedir, "seg_dir2")

    if not os.path.isdir(seg_dir1):
        os.mkdir(seg_dir1)
    if not os.path.isdir(seg_dir2):
        os.mkdir(seg_dir2)

    with open(raw_pickle_filepath, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        patient_dict = pickle.load(f)

    for patient_id in patient_dict:
        # patient_id is 0, 1, ..., N
        print(f"processing patient_id: {patient_id}")
        patient_uuid = patient_dict[patient_id]['id']
        if exclude_patient_ids is not None:
            if patient_uuid not in exclude_patient_ids:
                if mr_sequence in patient_dict[patient_id]:
                    # create patient directory
                    current_exportdir = os.path.join(export_basedir, patient_uuid)
                    current_exportdir_img = os.path.join(current_exportdir, 'T2W')
                    current_exportdir_seg = os.path.join(current_exportdir, 'GT')
                    current_exportdir_seg_cap = os.path.join(current_exportdir_seg, 'cap')
                    current_exportdir_seg_pro = os.path.join(current_exportdir_seg, 'prostate')
                    current_exportdir_seg_pz = os.path.join(current_exportdir_seg, 'pz')
                    current_exportdir_seg_cg = os.path.join(current_exportdir_seg, 'cg')
                    current_exportdir_seg_cg_filt = os.path.join(current_exportdir_seg, 'cg_filt')
                    current_exportdir_seg_cg_raw = os.path.join(current_exportdir_seg, 'cg_raw')
                    current_exportdir_seg_cg_con = os.path.join(current_exportdir_seg, 'cg_con')

                    dir_list = [current_exportdir, current_exportdir_img, current_exportdir_seg,
                                current_exportdir_seg_cap, current_exportdir_seg_pro,
                                current_exportdir_seg_pz, current_exportdir_seg_cg,
                                current_exportdir_seg_cg_raw, current_exportdir_seg_cg_con,current_exportdir_seg_cg_filt]

                    print(f"export dir: {current_exportdir}")

                    for path in dir_list:
                        try:
                            os.mkdir(path)
                        except OSError as error:
                            print(error)

                    for img_idx in range(patient_dict[patient_id][mr_sequence]['img'].shape[0]):
                        # img_idx is the index of the current image in the sequence of the patient patient_id

                        # write image
                        # castFilter = sitk.CastImageFilter()
                        # castFilter.SetOutputPixelType(sitk.sitkInt16)
                        # Convert floating type image (imgSmooth) to int type (imgFiltered)
                        img_buf = patient_dict[patient_id][mr_sequence]['img'][img_idx]
                        img = sitk.GetImageFromArray(img_buf)
                        # img = castFilter.Execute(img)
                        sitk.WriteImage(img, os.path.join(current_exportdir_img, f"Image{img_idx}.dcm"))

                        # write segmentations
                        # cap / les
                        img = sitk.GetImageFromArray(patient_dict[patient_id][mr_sequence]['LES'][img_idx] * 255)
                        sitk.WriteImage(img, os.path.join(current_exportdir_seg_cap, f"Image{img_idx}.{fileext}"))
                        # pz / PZ
                        img_pz = sitk.GetImageFromArray(patient_dict[patient_id][mr_sequence]['PZ'][img_idx] * 255)
                        sitk.WriteImage(img_pz, os.path.join(current_exportdir_seg_pz, f"Image{img_idx}.{fileext}"))
                        # prostate / PRO
                        img_pro = sitk.GetImageFromArray(patient_dict[patient_id][mr_sequence]['PRO'][img_idx] * 255)
                        sitk.WriteImage(img_pro, os.path.join(current_exportdir_seg_pro, f"Image{img_idx}.{fileext}"))
                        # create the central gland class here

                        wg_map = 255 * patient_dict[patient_id][mr_sequence]['PRO'][img_idx]
                        pz_map = 255 * patient_dict[patient_id][mr_sequence]['PZ'][img_idx]
                        cg_map = wg_map - pz_map

                        img_cg_raw = sitk.GetImageFromArray(cg_map)
                        sitk.WriteImage(img_cg_raw,
                                        os.path.join(current_exportdir_seg_cg_raw, f"Image{img_idx}.{fileext}"))

                        # median filter cg_map
                        # kernel_size = 15 works well
                        # mean filter
                        alpha = 0.8
                        kernel_size = int(alpha * (np.sqrt(np.sum(cg_map / np.max(cg_map)) / np.pi)))

                        if kernel_size % 2 == 0:
                            kernel_size -= 1

                        print(f"img_idx: {img_idx}, kernel_size: {kernel_size}")

                        cg_map = medfilt2d(cg_map, kernel_size=kernel_size)

                        img_cg = sitk.GetImageFromArray(cg_map)
                        sitk.WriteImage(img_cg,
                                        os.path.join(current_exportdir_seg_cg_filt, f"Image{img_idx}.{fileext}"))

                        cg_map = np.array(cg_map - ((cg_map / 255) * (pz_map / 255)) * 255, dtype=np.uint8)

                        # find all contours and select the max contour
                        # sometimes the pc contains more than two contours extract the bigger one
                        pz_cnt, pz_hierarchy = cv2.findContours(pz_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # we have a pz contour otherwise cg_map is cg_map
                        if len(pz_cnt) > 0:
                            pz_cnt_max_idx = 0
                            pz_cnt_s = pz_cnt[0].shape[0]
                            for idx, pc in enumerate(pz_cnt):
                                if pc.shape[0] > pz_cnt_s:
                                    pz_cnt_s = pc.shape[0]
                                    pz_cnt_max_idx = idx

                            M = cv2.moments(pz_cnt[pz_cnt_max_idx])

                            pz_cx = int(M['m10'] / M['m00'])
                            pz_cy = int(M['m01'] / M['m00'])

                            pz_center = (pz_cx, pz_cy)
                            # Radius of circle
                            radius = 2

                            # Blue color in BGR
                            pz_color = (0, 255, 0)
                            cg_color = (255, 0, 0)

                            cg_map_export = cv2.cvtColor(cg_map.copy(), cv2.COLOR_GRAY2BGR)
                            cv2.circle(cg_map_export, pz_center, radius, pz_color, cv2.FILLED)

                            # find all contours and select the max contour
                            contours, hierarchy = cv2.findContours(cg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            # select the contour with topmost / smallest y coordinate
                            contours_centre_distance = np.zeros(len(contours))
                            contours_area = np.zeros_like(contours_centre_distance)
                            for cnt_idx, cnt in enumerate(contours):
                                # contour moments
                                M = cv2.moments(cnt)
                                if M['m00'] == 0:
                                    contours_area[cnt_idx] = M['m00']
                                    contours_centre_distance[cnt_idx] = np.sqrt(pz_cx ** 2 + pz_cy ** 2)
                                else:
                                    # contour coordinates
                                    cx = int(M['m10'] / M['m00'])
                                    cy = int(M['m01'] / M['m00'])
                                    cv2.circle(cg_map_export, (cx, cy), radius, cg_color, cv2.FILLED)
                                    cv2.line(cg_map_export, pz_center, (cx, cy), (0, 0, 255), 1)

                                    contours_centre_distance[cnt_idx] = np.sqrt((pz_cx - cx) ** 2 + (pz_cy - cy) ** 2)
                                    contours_area[cnt_idx] = M['m00']



                            # contour segmentations with moments
                            cv2.imwrite(os.path.join(current_exportdir_seg_cg_con, f"Image{img_idx}.{fileext}"),
                                        cg_map_export)

                            contour_max_area_idx = np.argmax(contours_area)
                            # select contour with smallest distance b/w pz centre and cg centres
                            contour_top_idx = np.argmin(contours_centre_distance)

                            if contours_area[contour_top_idx] >= 0.4 * contours_area[contour_max_area_idx]:
                                contour_top = contours[contour_top_idx]
                            else:
                                contour_top = contours[contour_max_area_idx]

                            # contour_top = contours[contour_top_idx]

                            cg_map = np.zeros_like(cg_map)
                            cv2.drawContours(cg_map, [contour_top], -1, color=(255, 255, 255), thickness=cv2.FILLED)

                        # or wavelet filter: the current wavelet configuration is set to an image size of 320 x 320
                        # if the sizes mismatch the wavelets have to be adapted
                        # cg_map = img_as_ubyte(wavelet_filtering(cg_map))
                        # add noise reduction here
                        # max value of cg_map is already 255
                        img_cg = sitk.GetImageFromArray(cg_map)
                        sitk.WriteImage(img_cg, os.path.join(current_exportdir_seg_cg, f"Image{img_idx}.{fileext}"))

                        if create_preview:
                            # create an image to check how well the cg creation worked
                            seg_map = np.concatenate(
                                (np.expand_dims(patient_dict[patient_id][mr_sequence]['BG'][img_idx], axis=0),
                                 np.expand_dims(patient_dict[patient_id][mr_sequence]['PZ'][img_idx], axis=0),
                                 np.expand_dims(cg_map / 255, axis=0),
                                 np.expand_dims(patient_dict[patient_id][mr_sequence]['PRO'][img_idx],
                                                axis=0)),
                                axis=0)

                            # save each image to disk
                            save_name1 = os.path.join(seg_dir1, f"{patient_id}_{img_idx}.png")
                            save_name2 = os.path.join(seg_dir2, f"{patient_id}_{img_idx}.png")

                            plot_segmentation2_I2CVB(image=img_buf, segmentation=seg_map,
                                                     include_classes=['bg', 'pz', 'cg', 'prostate'],
                                                     show_ticks=False, title=f"{patient_uuid}_{img_idx}",
                                                     save_name=save_name1,
                                                     save_only=True)

                            plot_segmentation2_I2CVB(image=img_buf, segmentation=seg_map[[0, 1, 3]],
                                                     include_classes=['bg', 'pz', 'prostate'],
                                                     show_ticks=False, title=f"{patient_uuid}_{img_idx}",
                                                     save_name=save_name2,
                                                     save_only=True)


if __name__ == '__main__':
    pass


