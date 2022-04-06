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

from ds_ude_mitk_process import ude_to_dicom
import ds_paths as dp

# Dataset
# Dataset file path
ds_dict = {
    # dataset filepath
    'fp': "uninitialized/path",
    # 'fp': "/home/user/projects/data/Siemens-3t/Siemens-3t",
    # the dataset samples have at least one of the following classes in the list M ust I nclude C lasses
    'mic': ['pz', 'cg', 'prostate'],
    # include the following classes in the evaluation I nclude C lasses E valuation
    'ice': ['bg', 'pz', 'cg', 'prostate']
}

target_one_hot = True

if __name__ == "__main__":
    # we only convert the ude files to dicoms and add the rest of the DsI2CVB data
    # the ds_ude_mitk_process.py generates from the raw ude data a pickle file with all data in it. this is the path
    # to that file
    pickle_filepath_1 = '/path/to/ude_dataset_pro.pickle'

    export_basedir = dp.ds_comb_l
    ude_to_dicom(raw_pickle_filepath=pickle_filepath_1, export_basedir=export_basedir, mr_sequence='T2',
                 create_preview=False)
