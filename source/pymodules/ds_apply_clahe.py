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
import numpy as np
import cv2
import SimpleITK as sitk
from DsTransformations import feature_scaling
import matplotlib.pyplot as plt
import warnings

if __name__ == "__main__":
    # https://docs.opencv.org/3.4/d6/db6/classcv_1_1CLAHE.html
    # Sets size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles.
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    override = False
    plot_img = False
    basedir = '/start/dir/with/dataset/in/it'
    warnings.warn(f"this overrides all files in: {basedir}, set override to True to continue")

    if override:
        list_of_files = []
        for (dirpath, dirnames, filenames) in os.walk(basedir):
            if os.path.basename(dirpath) == 'T2W':
                for filename in filenames:
                    if filename.endswith('.dcm'):
                        list_of_files.append(os.sep.join([dirpath, filename]))

        # apply the clahe filter
        for fname in list_of_files:
            print(fname)
            print(os.path.basename(fname))
            img = sitk.ReadImage(fname)
            # array must be of shape h x w
            arr = np.array(feature_scaling(sitk.GetArrayFromImage(img)) * 255, dtype=np.uint8).squeeze()

            equalized = clahe.apply(arr)
            # normal equalisation
            img_cg = sitk.GetImageFromArray(equalized)
            sitk.WriteImage(img_cg, fname)

    if plot_img:
        # testing
        fname = '/path/to/dicom/image'
        savename = '/path/to/savename/image.png'
        img = sitk.ReadImage(fname)
        # array must be of shape h x w
        arr = np.array(feature_scaling(sitk.GetArrayFromImage(img)) * 255, dtype=np.uint8).squeeze()
        arr_hist = cv2.calcHist([arr], [0], None, [255], [0, 255])
        # normalize the histogram
        arr_hist /= arr_hist.sum()
        arr_hist *= 100
        # normal equalisation
        equ = cv2.equalizeHist(arr)
        equ_hist = cv2.calcHist([equ], [0], None, [255], [0, 255])
        equ_hist /= equ_hist.sum()
        equ_hist *= 100

        # clahe
        equ_clahe = clahe.apply(arr)
        equ_clahe_hist = cv2.calcHist([equ_clahe], [0], None, [255], [0, 255])
        equ_clahe_hist /= equ_clahe_hist.sum()
        equ_clahe_hist *= 100
        # plot the normalized histograms
        plot_title = "Histogram normalisation"
        title_fontsize = 16
        ax_fontsize = 14

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(plot_title, fontsize=title_fontsize)

        ax1.imshow(arr, cmap="gray")
        ax1.set_title(f"original", fontsize=ax_fontsize)
        ax1.axis('off')
        ax2.plot(arr_hist, color="tab:red", linewidth=2)
        ax2.set_title(f"original", fontsize=ax_fontsize)
        ax2.set_xlabel("bins", fontsize=ax_fontsize)
        ax2.set_ylabel("percentage of pixels", fontsize=ax_fontsize)
        ax2.grid()

        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(plot_title, fontsize=title_fontsize)

        ax1.imshow(equ, cmap="gray")
        ax1.set_title(f"histogram equalisation", fontsize=ax_fontsize)
        ax1.axis('off')
        ax2.plot(equ_hist, color="tab:orange", linewidth=2)
        ax2.set_title(f"histogram equalisation", fontsize=ax_fontsize)
        ax2.set_xlabel("bins", fontsize=ax_fontsize)
        ax2.set_ylabel("percentage of pixels", fontsize=ax_fontsize)
        ax2.grid()

        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(plot_title, fontsize=title_fontsize)
        ax1.imshow(equ_clahe, cmap="gray")
        ax1.set_title(f"CLAHE", fontsize=ax_fontsize)
        ax1.axis('off')
        ax2.plot(equ_clahe_hist, color="tab:blue", linewidth=2)
        ax2.set_title(f"CLAHE", fontsize=ax_fontsize)
        ax2.set_xlabel("iteration", fontsize=ax_fontsize)
        ax2.set_ylabel("percentage of pixels", fontsize=ax_fontsize)
        ax2.grid()

        plt.tight_layout()
        plt.show()

        all = np.concatenate((arr, equ, equ_clahe), axis=1)
        plt.imshow(all, cmap="gray")
        plt.axis('off')
        plt.show()
