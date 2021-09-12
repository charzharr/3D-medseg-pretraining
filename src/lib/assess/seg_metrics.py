""" lib/assess/metrics3d.py  (Author: Charley Zhang, 2021)
Most commonly used segmentation metrics (or the ones I need for med-im projects.
Huge help from MONAI: https://www.github.com/Project-MONAI/MONAI

Usage Philosophy:
    - Can take either detached cpu tensors or numpy arrays
    - Returns all information (per example & per class results). It is up to
        functions lower in the stack to aggregate in the form the application
        requires. 
            e.g. batch_confusion_matrix returns a BxCx4

List of Metrics:
    - Confusion Matrix (2D & 3D, Classif & Seg)
    - Dice (2D & 3D, Seg)
    - Jaccard (2D & 3D, Seg)
    - Hausdorff (2D & 3D, Seg)
"""

import warnings
from collections import namedtuple

import numpy as np
import torch

from dmt.metrics.medpy_metrics import dc, jc, hd 
from .unify import (reshape, stack, to_float, to_int, allclose, nan_to_num,
                    any as uni_any, sum as uni_sum)
from .seg_utils import get_mask_edges, get_surface_distance


# Module scope for multiprocessing compatibility
SegCM = namedtuple('SegCM', ('tp', 'fp', 'fn', 'tn'))
Mets = namedtuple('Mets', ('confusion', 'dice', 'jaccard', 'exists'))


def batch_metrics(preds, targs, ignore_background=False, naive_avg=True):
    """
    Args:
        preds (tensor or array): BxCxDxHxW (binary one-hot)
        targs: (tensor or array) BxCxDxHxW (binary one-hot)
        naive_avg: just set to False for inference. May skew results if 
            a given pred & targ batch doesn't have all classes.
    """
    assert preds.shape == targs.shape
    assert preds.ndim == 5
    assert 'int' in str(preds.dtype)
    assert 'int' in str(targs.dtype)
    
    cdj_tuple = batch_cdj_metrics(preds, targs,
                                  ignore_background=ignore_background)

    nt_conf = cdj_tuple.confusion  # named_tuple: tp, fp, tn, fn
    nt_conf = SegCM(np.array(nt_conf.tp), np.array(nt_conf.fp), 
                 np.array(nt_conf.fn), np.array(nt_conf.tn))

    ec_dice = np.array(cdj_tuple.dice)  # ec = element class
    ec_jaccard = np.array(cdj_tuple.jaccard)
    ec_exists = np.array(cdj_tuple.exists)

    with np.errstate(divide='ignore', invalid='ignore'):
        if naive_avg:
            dice_class = ec_dice.mean(0)  # shape=(C,)
            dice_batch = ec_dice.mean(1)
            dice_mean = dice_batch.mean()
            jaccard_class = ec_jaccard.mean(0)
            jaccard_batch = ec_jaccard.mean(1)
            jaccard_mean = jaccard_batch.mean()
        else:
            dice_class = ec_dice.sum(0) / ec_exists.sum(0)
            dice_batch = ec_dice.sum(1) / ec_exists.sum(1)
            jaccard_class = ec_jaccard.sum(0) / ec_exists.sum(0)
            jaccard_batch = ec_jaccard.sum(1) / ec_exists.sum(1)
            
            for a in (dice_class, dice_batch, jaccard_class, jaccard_batch):
                a[a == np.inf] = np.nan
            batch_exist_count = np.sum(~np.isnan(dice_batch))
            dice_mean = dice_batch.sum() / batch_exist_count
            dice_mean = np.nan if dice_mean == np.inf else dice_mean
            jaccard_mean = jaccard_batch.sum() / batch_exist_count
            jaccard_mean = np.nan if jaccard_mean == np.inf else jaccard_mean

    return {
        'confusion_all': nt_conf,  # named tuple
        'dice_all': ec_dice,
        'dice_class': dice_class,  # shape=(C,)
        'dice_batch': dice_batch,  # shape=(B,)
        'dice_mean': dice_mean,
        'jaccard_all': ec_jaccard,
        'jaccard_class': jaccard_class,  # shape=(C,)
        'jaccard_batch': jaccard_batch,  # shape=(B,)
        'jaccard_mean': jaccard_mean,
    }




def batch_cdj_metrics(pred, targ, ignore_background=True):
    """ Optimized execution to get Confusion Matrix, Dice, and Jaccard.
    Args:
        pred: BxC(xD)xHxW tensor or array
        targ: BxC(xD)xHxW tensor or array
        ignore_background (bool): flag to ignore first channel dim or not
    Returns:
        A namedtuple that has the following keys:
            - 'confusion' a namedtuple that has tp, fp, fn, tn arrays
            - 'jaccard' a BxC array of ious (nans are turned into 0s)
            - 'dice' a BxC array of dice scores (nans are turned into 0s)
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    CM = batch_confusion_matrix(pred, targ, ignore_background=ignore_background)
    tp = CM.tp  # BxC
    fp = CM.fp
    fn = CM.fn
    exists = to_float((tp + fn) > 0)  # if b, c has item, then 1
    
    # Get Dice & Jaccard
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)  # BxC
    # dice = nan_to_num(dice, nan=0)
    # dice_sanity = batch_dice(pred, targ)
    # assert allclose(dice, dice_sanity)

    jaccard = tp / (tp + fp + fn + 1e-8)  # BxC
    # jaccard = nan_to_num(jaccard, nan=0)
    # jaccard_sanity = batch_jaccard(pred, targ)
    # assert allclose(jaccard, jaccard_sanity)

    return Mets(CM, dice, jaccard, exists)


# ------------ ##  Individual Metrics: CM, Dice, Jaccard  ## ----------- # 

def batch_confusion_matrix(pred, targ, ignore_background=True):
    """ 2D or 3D image for segmenation. 
    Args:
        pred: BxC(xD)xHxW tensor or array
        targ: BxC(xD)xHxW tensor or array
        ignore_background (bool): flag to ignore first channel dim or not
    Returns:
        BxCx4 (Batch x Classes x TP,FP,TN,FN
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
#     if isinstance(pred, torch.Tensor):
#         pred = pred.float()
#         targ = targ.float()
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ

    # Flatten pred & targ to B x C x S (S = all pixels for seg, 1 for classif)
    B, C = targ.shape[:2]
    pred_flat = reshape(pred, (B, C, -1))
    targ_flat = reshape(targ, (B, C, -1))
    
    tp = to_int((pred_flat + targ_flat) == 2)  # BxCxS
    tn = to_int((pred_flat + targ_flat) == 0)  # BxCxS

    tp = uni_sum(tp, axis=[2])  # BxC
    tn = uni_sum(tn, axis=[2])  # BxC
    
    p = uni_sum(targ_flat, axis=[2])  # BxC, count of all positives
    n = pred_flat.shape[-1] - p  # BxC, count of all negatives

    fn = p - tp
    fp = n - tn

    return SegCM(tp, fp, fn, tn)  # can be int or long if inputs are int


def batch_dice(pred, targ, ignore_background=True):
    """
    Args:
        pred: BxCxDxHxW binary array
        targ: BxCxDxHxW binary array
    """    
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ
    
    B, C = targ.shape[0], targ.shape[1]
    if isinstance(pred, np.ndarray):
        elem_class_dice = np.zeros((B, C)).astype(np.float32)
    else:
        elem_class_dice = torch.zeros((B, C), dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            elem_class_pred = pred[b, c]
            elem_class_targ = targ[b, c]
            
            intersection = np.count_nonzero(elem_class_pred & elem_class_targ)
            pred_area = np.count_nonzero(elem_class_pred)
            targ_area = np.count_nonzero(elem_class_targ)

            denom = pred_area + targ_area
            if denom == 0:
                elem_class_dice[b, c] = 0.
            else:
                elem_class_dice[b, c] = 2. * intersection / denom
    
    return elem_class_dice


def batch_jaccard(pred, targ, ignore_background=True):
    """
    Args:
        pred: BxCxDxHxW binary array
        targ: BxCxDxHxW binary array
    """    
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ
    
    B, C = targ.shape[0], targ.shape[1]
    if isinstance(pred, np.ndarray):
        elem_class_jaccard = np.zeros((B, C)).astype(np.float32)
    else:
        elem_class_jaccard = torch.zeros((B, C), dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            elem_class_pred = pred[b, c]
            elem_class_targ = targ[b, c]
            
            intersection = np.count_nonzero(elem_class_pred & elem_class_targ)
            pred_area = np.count_nonzero(elem_class_pred)
            targ_area = np.count_nonzero(elem_class_targ)
            union = pred_area + targ_area - intersection
            
            if union <= 0:
                elem_class_jaccard[b, c] = 0.
            else:
                elem_class_jaccard[b, c] = intersection / union
    
    return elem_class_jaccard


# ------------ ##  Hausdorff Metric & Helpers  ## ----------- # 

def batch_hausdorff(
        pred, 
        targ, 
        ignore_background=True,
        distance_metric='euclidean',
        percentile=None,
        directed=False
        ):
    """ 
    Args: 
        pred: BxCxDxHxW binary array or tensor
        targ: BxCxDxHxW binary array or tensor
        ignore_background: flag to take out 1st class dimension or not
        distance_metric: 'euclidean', 'chessboard', 'taxicab'
        percentile: [0, 100], return percentile of distance rather than max.
        directed: flag to calculated directed Hausdorff distance or not.
    """
    assert type(pred) == type(targ), f'Types: {type(pred)}, {type(targ)}'
    assert isinstance(pred, np.ndarray) or isinstance(pred, torch.Tensor)
    assert pred.shape == targ.shape, f'{pred.shape} {targ.shape} mismatch!'
    
    if ignore_background:
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        targ = targ[:, 1:] if targ.shape[1] > 1 else targ

    B, C = targ.shape[:2]
    if isinstance(pred, np.ndarray):
        HD = np.zeros((B, C)).astype(np.float32)
    else:
        HD = torch.zeros((B, C), dtype=torch.float32)
    
    for b, c in np.ndindex(B, C):
        (edges_pred, edges_gt) = get_mask_edges(pred[b, c], targ[b, c])
        if not uni_any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan/inf distance.")
        if not uni_any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan/inf distance.")

        distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile)
        if directed:
            HD[b, c] = distance_1
        else:
            distance_2 = compute_percent_hausdorff_distance(edges_gt, edges_pred, distance_metric, percentile)
            HD[b, c] = max(distance_1, distance_2)
    return torch.from_numpy(hd)




# ------------ ##  Rudimentary Tests  ## ----------- # 

if __name__ == '__main__':
    
    pred_array = np.random.randint(0, 2, size=(2, 3, 5, 5))
    targ_array = np.random.randint(0, 2, size=(2, 3, 5, 5))
    
    pred_tens = torch.tensor(pred_array)
    targ_tens = torch.tensor(targ_array)
    
    # Confusion Matrix Tests
    cm_tens = batch_confusion_matrix(pred_tens, targ_tens)
    cm_array = batch_confusion_matrix(pred_array, targ_array)
    
    assert allclose(cm_tens.numpy(), cm_array)
    
    # CM Package
    cdj_tens = batch_cdj_metrics(pred_tens, targ_tens)
    cdj_array = batch_cdj_metrics(pred_array, targ_array)
    assert allclose(cdj_tens.confusion.numpy(), cdj_array.confusion)
    assert allclose(cdj_tens.dice.numpy(), cdj_array.dice)
    assert allclose(cdj_tens.jaccard.numpy(), cdj_array.jaccard)
    