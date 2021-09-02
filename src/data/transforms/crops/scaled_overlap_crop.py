""" scaled_overlap_crop.py (Author: Charley Zhang, 2021)
A transform that only works on 3D tensors & np arrays. It's used to reimplement
the PGL overlapping crop feature needed for contrastive pretraining.

PGL Transformations (2 spatial, 4 intensity): 
- Crop & Scale (110% to 140% of final crop size), at least 10% overlap
- 50% flip along x, y, z axis
- Gaussian noise (uniform from 0 to 0.1 variance), p=0.1
- Gaussian blur (sigma=[0.5, 1]) p=0.2
- Brightness / contrast (1st mult by [0.75, 1.25] then clipped), p=0.5
- Gamma transform (λ=[0.7,1.5], then scaled to [0,1]), p=0.5

Reversal for MSE calculation:
- First reverse axis flipping
- Then reverse the crop/scaling
"""

import warnings
import math
from cv2 import resize
import numpy as np
import torch

if __name__ == '__main__':
    import sys, pathlib
    curr_path = pathlib.Path().absolute()
    sys.path.append(str(curr_path.parent.parent.parent))

from lib.utils.parse import parse_positive_int
from data.transforms import Resize3d
from data.transforms.crops.utils import sample_crop


class ScaledOverlapCropper3d:
    """
    Input: tensor or np array
    
    Execution Walkthrough
        1. Given a full volume, 1 random crop of size(scale * final_shape)
            is sampled. 
        2. Using the min_overlap parameter & a random valid corner to which
            the 2nd crop will overlap, the 2nd crop is sampled.
            Note: shape of 2nd crop is independent of 1st & sampled in the 
                same way using the scale_range & final_shape.
        3. Both crops are resized to the final_shape & their overlap 
            coordinates are adjusted.
        4. The overlapping crop pair is returned with their meta info. 
    
    PGL Reimplementation Assumptions
    - scale sample applies to each axis equally, not each axis samples own scale
    - only j & k axis have variable scale, i axis has complete overlap always
    """
    def __init__(self, num_overlap_crops=2):
        self.num_overlap_crops = parse_positive_int(num_overlap_crops,
                                                    'num_overlap_crops')
    
    def __call__(
            self,
            volume_tensor,
            final_shape,
            scale_range=(1.1, 1.4),
            min_overlap=0.1,
            n_times=1,
            interpolation='trilinear'
            ):
        """
        Args:
            volume_tensor: DxHxW image tensor
            final_shape: final crop output shape
            scale_range: range of numbers to sample the scale of the initial
                volume before being resized to the final_shape
            min_overlap: percentage overlap between overlapping patch pairs
            n_times: number of patch pairs to return
        """
        assert len(scale_range) == 2
        assert 0 <= min_overlap <= 1
        n_times = parse_positive_int(n_times, 'n_times')
        
        vol_shape = volume_tensor.shape
        assert len(vol_shape) == len(final_shape)
        
        resize_transform = Resize3d(final_shape, return_record=False,
                                    default_interpolation=interpolation)
        
        ret = []
        for i in range(n_times):
            pair = self._get_overlapping_crops(volume_tensor, final_shape, 
                            scale_range, min_overlap, resize_transform)
            ret.append(pair)
        return ret
    
    
    def invert(crop_tensor, settings):
        """ TODO Differentiable inverting of scaling. """
        raise NotImplementedError()
    
    
    def _get_overlapping_crops(self, tensor, final_shape, scale_range, 
                               min_overlap, resize_transform):
        """ Called by __call__() to return overlapping crops as 1 example.
        1. Sample crop sizes for 2 crops.
        2. Sample crop 1 coordinates from size.
        3. Sample a valid corner.
        4. From crop2 size & valid corner of crop1, sample crop 2 coordinates.
        """
        tensor_shape = list(tensor.shape)
        final_shape = list(final_shape)
        
        # Sample crop sizes
        crop1_shape = [final_shape[0]] + [
            torch.randint(math.ceil(scale_range[0] * final_shape[i]), 
                          int(scale_range[1] * final_shape[i]), (1,)).item()
            for i in range(1, 3)
        ]
        coords1 = sample_crop(tensor_shape, crop1_shape)
        crop1_lower, crop1_upper = coords1
        crop1 = {'final_shape': final_shape,
                 'init_shape': crop1_shape,
                 'init_lower': list(crop1_lower),
                 'init_upper': list(crop1_upper)}  # exclusive range: [low, up)
        
        crop2_shape = [final_shape[0]] + [
            torch.randint(math.ceil(scale_range[0] * final_shape[i]), 
                          int(scale_range[1] * final_shape[i]), (1,)).item()
            for i in range(1, 3)
        ]
        crop2 = {'final_shape': final_shape,
                 'init_shape': crop2_shape}  
        
        #-----#  Sample overlap corner. 0=TL, 1=TR, 2=BR, 3=BL  #-----#
        # Calculate minimum area to satisfy overlap & crop H, W
        crop1_w, crop1_h = crop1_shape[2], crop1_shape[1]
        crop2_w, crop2_h = crop2_shape[2], crop2_shape[1]
        
        crop1_area = crop1_w * crop1_h
        crop2_area = crop2_w * crop2_h
        min_overlap_area = min_overlap * (crop1_area + 
                            crop2_area) / (1 + min_overlap)
            # Note: solve eq A_o / (A_1 + A_2 - A_o) = min_overlap
            #  where A_o = area of overlap -> min_overlap_area
        
        # Calculate the min vertical & horizontal lengths based on final_shape
        #  H/W ratios.
        min_vert_overlap = math.ceil(math.sqrt((final_shape[1] * 
                                        min_overlap_area) / final_shape[2]))
        min_hori_overlap = math.ceil(min_overlap_area / min_vert_overlap)
            # Note: old incorrect overlap calculation
            # min_vert_overlap = round(crop1_shape[1] * min_overlap)
            # min_hori_overlap = round(crop1_shape[2] * min_overlap)
        
        # Sample crop for either of 4 corners
        corners = list(range(1, 5))
        while True:
            corner = corners[int(torch.rand((1,)).item() * len(corners))]
                
            if corner == 1:  # top left
                y_min = crop1_lower[1] + min_vert_overlap - crop2_h
                x_min = crop1_lower[2] + min_hori_overlap - crop2_w
                if y_min < 0 or x_min < 0:
                    # print(f'Corner {corner} failed!')
                    # print(f'Crop 1 shape: {crop1_shape}')
                    # print(f'Crop 1 lower: {crop1_lower}, upper: {crop1_upper}')
                    # print(f'Min vert overlap {min_vert_overlap}, '
                    #       f'min hori overlap {min_hori_overlap}')
                    # print(f'y_min {y_min} x_min {x_min}')
                    # import sys; sys.exit(1)
                    corners.remove(corner)
                    continue
                y_max = min(crop1_upper[1] - crop2_h, y_min + crop2_h)
                x_max = min(crop1_upper[2] - crop2_w, x_min + crop2_w)
            elif corner == 2:  # top right
                y_min = crop1_lower[1] + min_vert_overlap - crop2_h
                x_max = crop1_upper[2] - min_hori_overlap + crop2_w
                if y_min < 0 or x_max > tensor_shape[2]:
                    # print(f'Corner {corner} failed!')
                    # print(f'Crop 1 shape: {crop1_shape}')
                    # print(f'Crop 1 lower: {crop1_lower}, upper: {crop1_upper}')
                    # print(f'Min vert overlap {min_vert_overlap}, '
                    #       f'min hori overlap {min_hori_overlap}')
                    # print(f'y_min {y_min} x_max {x_max}')
                    # import sys; sys.exit(1)
                    corners.remove(corner)
                    continue
                x_max = crop1_upper[2] - min_hori_overlap
                x_min = max(crop1_lower[2], x_max - crop2_w)
                y_max = min(crop1_upper[1] - crop2_h, y_min + crop2_h)
            elif corner == 3:  # bottom right
                y_max = crop1_upper[1] - min_vert_overlap + crop2_h
                x_max = crop1_upper[2] - min_hori_overlap + crop2_w
                if y_max > tensor_shape[1] or x_max > tensor_shape[2]:
                    # print(f'Corner {corner} failed!')
                    # print(f'Crop 1 shape: {crop1_shape}')
                    # print(f'Crop 1 lower: {crop1_lower}, upper: {crop1_upper}')
                    # print(f'Min vert overlap {min_vert_overlap}, '
                    #       f'min hori overlap {min_hori_overlap}')
                    # print(f'y_max {y_max} x_max {x_max}')
                    # import sys; sys.exit(1)
                    corners.remove(corner)
                    continue
                y_max = crop1_upper[1] - min_vert_overlap
                y_min = max(crop1_lower[1], y_max - crop2_h)
                x_max = crop1_upper[2] - min_hori_overlap
                x_min = max(crop1_lower[2], x_max - crop2_w)
            else:  # bottom left
                y_max = crop1_upper[1] - min_vert_overlap + crop2_h
                x_min = crop1_lower[2] + min_hori_overlap - crop2_w
                if y_max > tensor_shape[1] or x_min < 0:
                    # print(f'Corner {corner} failed!')
                    # print(f'Crop 1 shape: {crop1_shape}')
                    # print(f'Crop 1 lower: {crop1_lower}, upper: {crop1_upper}')
                    # print(f'Min vert overlap {min_vert_overlap}, '
                    #       f'min hori overlap {min_hori_overlap}')
                    # print(f'y_max {y_max} x_min {x_min}')
                    # import sys; sys.exit(1)
                    corners.remove(corner)
                    continue
                y_max = crop1_upper[1] - min_vert_overlap
                y_min = max(crop1_lower[1], y_max - crop2_h)
                x_max = min(crop1_upper[2] - crop2_w, x_min + crop2_w)
            break
                
        # Sample crop2 position via x/y_min/max
        # print(corner, x_min, x_max, y_min, y_max)
        # print('min vert overlap', min_vert_overlap)
        # print('min hori overlap', min_hori_overlap)
        
        crop2_lower = [crop1_lower[0]] + [
            torch.randint(y_min, y_max, (1,)).item(),
            torch.randint(x_min, x_max, (1,)).item()
        ]
        crop2_upper = [crop2_lower[i] + crop2_shape[i] for i in range(3)]
        crop2['init_lower'] = crop2_lower
        crop2['init_upper'] = crop2_upper
        
        # Record overlap metadata & iou
        overlap_area = ScaledOverlapCropper3d.get_crop_overlap_area(crop1_lower, 
                    crop1_upper, crop2_lower, crop2_upper)
        assert overlap_area > 0, 'WTF'
        
        overlap_d = ScaledOverlapCropper3d.get_crop_overlap(crop1_lower, 
                    crop1_upper, crop2_lower, crop2_upper)
        overlap_d['init_iou'] = get_iou((crop1_lower[2], crop1_lower[1], 
                                         crop1_upper[2], crop1_upper[1]), 
                                        (crop2_lower[2], crop2_lower[1], 
                                         crop2_upper[2], crop2_upper[1]))
        crop1['init_overlap'] = overlap_d
        crop2['init_overlap'] = overlap_d
        
        # Resize & final tensor metadata
        crop1_tens = ScaledOverlapCropper3d.crop(tensor, crop1_lower, crop1_upper)
        crop1_tens = resize_transform(crop1_tens)
        crop1['final_tensor'] = crop1_tens
        
        crop2_tens = ScaledOverlapCropper3d.crop(tensor, crop2_lower, crop2_upper)
        crop2_tens = resize_transform(crop2_tens)
        crop2['final_tensor'] = crop2_tens
        
        dim_ratios1 = [f/i for f, i in zip(final_shape, crop1_shape)]
        l_overlap_rel_crop1 = [o - c for c, o in zip(crop1_lower, 
                                                     overlap_d['lower'])]
        u_overlap_rel_crop1 = [o - c for c, o in zip(crop1_lower, 
                                                     overlap_d['upper'])]
        crop1['final_relative_overlap'] = {
            'lower': [r * i for r, i in zip(dim_ratios1, l_overlap_rel_crop1)],
            'upper': [r * i for r, i in zip(dim_ratios1, u_overlap_rel_crop1)],
        }
        
        dim_ratios2 = [f/i for f, i in zip(final_shape, crop2_shape)]
        l_overlap_rel_crop2 = [o - c for c, o in zip(crop2_lower, 
                                                     overlap_d['lower'])]
        u_overlap_rel_crop2 = [o - c for c, o in zip(crop2_lower, 
                                                     overlap_d['upper'])]
        crop2['final_relative_overlap'] = {
            'lower': [r * i for r, i in zip(dim_ratios2, l_overlap_rel_crop2)],
            'upper': [r * i for r, i in zip(dim_ratios2, u_overlap_rel_crop2)],
        }
        
        return crop1, crop2
    
    
    @staticmethod
    def crop(tensor, lower, upper):
        return tensor[lower[0]:upper[0], 
                      lower[1]:upper[1], 
                      lower[2]:upper[2]].clone()
    
    
    @staticmethod
    def get_crop_overlap_area(lower1, upper1, lower2, upper2):
        axis_overlap = []
        for i in range(3):
            if lower1[i] < lower2[i]:
                axis_overlap.append(min(upper2[i], upper1[i]) - lower2[i])
            else:
                axis_overlap.append(min(upper2[i], upper1[i]) - lower1[i])
        overlap = axis_overlap[0] * axis_overlap[1] * axis_overlap[2]
        return overlap if overlap > 0 else 0
    
    
    @staticmethod
    def get_crop_overlap(lower1, upper1, lower2, upper2):
        lower, upper = [], []
        for i in range(3):
            if lower1[i] < lower2[i]:
                upper.append(min(upper2[i], upper1[i]))
                lower.append(lower2[i])
                if upper[-1] - lower[-1] < 0:
                    warnings.warn(f'No overlap in {i+1}th dimension')
                    return None
            else:
                upper.append(min(upper2[i], upper1[i]))
                lower.append(lower1[i])
                if upper[-1] - lower[-1] < 0:
                    warnings.warn(f'No overlap in {i+1}th dimension')
                    return None
        return {
            'lower': lower,
            'upper': upper,
            'shape': [u - l for u, l in zip(upper, lower)]
        }


def get_iou(bb1, bb2):
    """ Gets 2D IoU. """
    bb1 = {'x1': bb1[0], 'y1': bb1[1], 'x2': bb1[2], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'y1': bb2[1], 'x2': bb2[2], 'y2': bb2[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou



if __name__ == '__main__':
    from lib.utils.io import output
    import matplotlib.pyplot as plt
    import cv2
    
    def print_crop(crop_d):
        for k, v in crop_d.items():
            if isinstance(v, torch.Tensor):
                print(f'{k}: {v.shape}')
            else:
                print(f'{k}: {v}')
                
    def draw_axial_crops(image, lower1, upper1, lower2, upper2):
        yt1, xt1 = int(lower1[1]), int(lower1[2])
        yb1, xb1 = int(upper1[1]), int(upper1[2])
        yt2, xt2 = int(lower2[1]), int(lower2[2])
        yb2, xb2 = int(upper2[1]), int(upper2[2])
        
        image = cv2.rectangle(image, (xt1, yt1), (xb1, yb1), (0, 0, 255))
        image = cv2.rectangle(image, (xt2, yt2), (xb2, yb2), (255, 0, 0))
        print(f'IoU: {get_iou((xt1, yt1, xb1, yb1), (xt2, yt2, xb2, yb2))}')
        return image
    
    soc = ScaledOverlapCropper3d()
    final_shape = (16, 96, 96)
    
    # --- # Visualize the crops # --- #
    visualize_overlap = True
    
    for i in range(1):
        output.subsection(f'Crop Set {i+1}')
        exs = soc(torch.randn((100, 512, 512)), final_shape, n_times=2)

        # continue
        for crop1, crop2 in exs:
            image = np.zeros((512, 512, 3))
            output.subsubsection(f'Crop Set {i+1} Pair')
            print(f'Crop 1')
            print_crop(crop1)
            print(f'Crop 2')
            print_crop(crop2)
            image = draw_axial_crops(image, 
                crop1['init_lower'], crop1['init_upper'], 
                crop2['init_lower'], crop2['init_upper'])
            
            if visualize_overlap:  # draw the overlap crop in white
                yt1, xt1 = crop2['init_overlap']['lower'][1:]
                yb1, xb1 = crop2['init_overlap']['upper'][1:]
                color = (255, 255, 255)
                image = cv2.rectangle(image, (xt1, yt1), (xb1, yb1), color)
            
            fig = plt.figure(figsize=(18, 8))
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(image.clip(0, 255))
            
            crop1_image = np.zeros((96, 96, 3))
            crop1_y1, crop1_x1 = crop1['final_relative_overlap']['lower'][1:]
            crop1_y2, crop1_x2 = crop1['final_relative_overlap']['upper'][1:]
            crop1_image = cv2.rectangle(crop1_image, 
                                        (int(crop1_x1), int(crop1_y1)),
                                        (int(crop1_x2), int(crop1_y2)),
                                        (255, 255, 255), -1)
            ax = fig.add_subplot(1, 3, 2)
            ax.set_title('Final Resized Crop 1 & Overlap (white)')
            ax.imshow(crop1_image.clip(0, 255))
            
            crop2_image = np.zeros((96, 96, 3))
            crop2_y1, crop2_x1 = crop2['final_relative_overlap']['lower'][1:]
            crop2_y2, crop2_x2 = crop2['final_relative_overlap']['upper'][1:]
            crop2_image = cv2.rectangle(crop2_image, 
                                        (int(crop2_x1), int(crop2_y1)),
                                        (int(crop2_x2), int(crop2_y2)),
                                        (255, 255, 255), -1)
            ax = fig.add_subplot(1, 3, 3)
            ax.set_title('Final Resized Crop 2 & Overlap (white)')
            ax.imshow(crop2_image.clip(0, 255))
            plt.show()

            
        
        
        
        
        
        