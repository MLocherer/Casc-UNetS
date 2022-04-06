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

run_path_l = "/path/to/your/local/model/files/and/configuration/files"
run_path_r = "/path/to/your/remote/model/files/and/configuration/files"
# plain
# local
ds_base_path_l = "/path/to/your/local/nohe/datasets"
ds_i2cvb_l = os.path.join(ds_base_path_l, 'I2CVB', 'Siemens-3t')
ds_ude_l = os.path.join(ds_base_path_l, 'ude2')
ds_comb_l = os.path.join(ds_base_path_l, 'combined')
# remote paths
ds_base_path_r = "/path/to/your/remote/nohe/datasets"
ds_i2cvb_r = os.path.join(ds_base_path_r, 'I2CVB', 'Siemens-3t')
ds_ude_r = os.path.join(ds_base_path_r, 'ude2')
ds_comb_r = os.path.join(ds_base_path_r, 'combined')

# clahe
# local
ds_base_path_c_l = "/path/to/your/local/clahe/datasets"
ds_i2cvb_c_l = os.path.join(ds_base_path_c_l, 'I2CVB', 'Siemens-3t')
ds_ude_c_l = os.path.join(ds_base_path_c_l, 'ude')
ds_comb_c_l = os.path.join(ds_base_path_c_l, 'combined')
# remote paths
ds_base_path_c_r = "/path/to/your/remote/clahe/datasets"
ds_i2cvb_c_r = os.path.join(ds_base_path_c_r, 'I2CVB', 'Siemens-3t')
ds_ude_c_r = os.path.join(ds_base_path_c_r, 'ude2')
ds_comb_c_r = os.path.join(ds_base_path_c_r, 'combined')

ds_path_dict = {
    'local': {
        'rp': run_path_l,
        'plain': {
            'I2CVB': ds_i2cvb_l,
            'UDE': ds_ude_l,
            'combined': ds_comb_l
        },
        'clahe': {
            'I2CVB': ds_i2cvb_c_l,
            'UDE': ds_ude_c_l,
            'combined': ds_comb_c_l
        },
    },
    'remote': {
        'rp': run_path_r,
        'plain': {
            'I2CVB': ds_i2cvb_r,
            'UDE': ds_ude_r,
            'combined': ds_comb_r
        },
        'clahe': {
            'I2CVB': ds_i2cvb_c_r,
            'UDE': ds_ude_c_r,
            'combined': ds_comb_c_r
        }
    }
}
