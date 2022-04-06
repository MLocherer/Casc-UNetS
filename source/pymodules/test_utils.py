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
import cv2
import numpy as np
import torch
from skimage import img_as_ubyte


def filter_prostate_contours(seg_map, convex_hull=False):
    if torch.is_tensor(seg_map):
        seg_map = img_as_ubyte(seg_map.cpu().numpy())

    # find all contours and select the max contour
    contours, hierarchy = cv2.findContours(seg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # image centre
        centre_x, centre_y = seg_map.shape[0] / 2, seg_map.shape[1] / 2

        # select the contour with topmost / smallest y coordinate
        contours_centre_distance = np.zeros(len(contours))
        contours_area = np.zeros_like(contours_centre_distance)
        for cnt_idx, cnt in enumerate(contours):
            # contour moments
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                contours_centre_distance[cnt_idx] = np.sqrt(centre_x ** 2 + centre_y ** 2)
                contours_area[cnt_idx] = 0
            else:
                # contour coordinates
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                contours_centre_distance[cnt_idx] = np.sqrt((centre_x - cx) ** 2 + (centre_y - cy) ** 2)
                contours_area[cnt_idx] = M['m00']

        contour_max_area_idx = np.argmax(contours_area)

        # select the topmost contour
        contour_top_idx = np.argmin(contours_centre_distance)


        # prevent selection of very small contours with low y-value
        # .6 is to low and creates bad samples
        if contours_area[contour_top_idx] >= 0.05 * contours_area[contour_max_area_idx]:
            contour_top = contours[contour_top_idx]
        else:
            contour_top = contours[contour_max_area_idx]

        if convex_hull:
            # convex hull
            contour_top = cv2.convexHull(contour_top)

        seg_map = np.zeros_like(seg_map)
        cv2.drawContours(seg_map, [contour_top], -1, color=(255, 255, 255), thickness=cv2.FILLED)

    return torch.from_numpy(seg_map / 255)


def count_contours(seg_map, channels):
    r"""
    counts the number of individual contours in the predicted seg_map of shape N x C x H x W
    Args:
        channels: list of Channel dimension to count contours for
        seg_map: segmentation map

    Returns:
        numpy array
    """
    if torch.is_tensor(seg_map):
        seg_map = img_as_ubyte(seg_map.cpu().numpy())

    total_contours = np.zeros(len(channels))

    for i, channel in enumerate(channels):
        for idx in range(seg_map.shape[0]):
            # find all contours and select the max contour
            contours, hierarchy = cv2.findContours(seg_map[idx][channel], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total_contours[i] += len(contours)

    return total_contours


def filter_cap(seg_map_cap, prostate_mask, convex_hull=True):
    r"""
    removes all predictions outside the prostate contour
    Args:
        seg_map_cap: mask to modify
        prostate_mask: mask to apply to seg_map_cap
        convex_hull: calculate convex hull of prostate_mask first
    Returns:

    """
    if torch.is_tensor(seg_map_cap):
        seg_map_cap = img_as_ubyte(seg_map_cap.cpu().numpy())
    if torch.is_tensor(prostate_mask):
        prostate_mask = img_as_ubyte(prostate_mask.cpu().numpy())

    if convex_hull:
        prostate_mask_c = np.zeros_like(prostate_mask)
        for idx in range(prostate_mask.shape[0]):
            contours, hierarchy = cv2.findContours(prostate_mask[idx], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                # select the contour with topmost / smallest y coordinate
                contours_area = np.zeros(len(contours))
                for cnt_idx, cnt in enumerate(contours):
                    # contour moments
                    M = cv2.moments(cnt)
                    if M['m00'] == 0:
                        contours_area[cnt_idx] = 0
                    else:
                        # contour coordinates
                        contours_area[cnt_idx] = M['m00']

                contour_max_area_idx = np.argmax(contours_area)
                contour_mask = cv2.convexHull(contours[contour_max_area_idx])

                cv2.drawContours(prostate_mask_c[idx], [contour_mask], -1, color=(255, 255, 255), thickness=cv2.FILLED)

    return torch.from_numpy(seg_map_cap * prostate_mask_c / 255).to(torch.uint8)


