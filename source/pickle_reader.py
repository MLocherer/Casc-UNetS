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

import pickle

if __name__ == "__main__":
    r"""
        This file just demonstrates the contents of a pickle file. A pickle file was created during training and
        contains all the information regarding the run. For each 5cv fold there exists a different model. There is
        a dict with called x with the following structure:
    """

    fname = 'filepath/to/pickle/file/unet1_combined_clahe_5cv.pickle'

    with open(fname, 'rb') as f:
        x = pickle.load(f)

    print('use debugger to view data')