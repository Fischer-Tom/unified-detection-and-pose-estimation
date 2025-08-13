import glob
import math
import os
import _pickle as cPickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def compute_RT_degree_cm_symmetry(
    RT_1, RT_2, class_id, handle_visibility, synset_names
):
    """
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    """

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if synset_names[class_id] in ["bottle", "can", "bowl"]:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif synset_names[class_id] == "mug" and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ["phone", "eggbox", "glue"]:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(
            np.arccos((np.trace(R) - 1) / 2), np.arccos((np.trace(R_rot) - 1) / 2)
        )
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = (
            np.array(
                [
                    [scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                    [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                    [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                    [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                    [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                    [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                    [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                    [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                ]
            )
            + shift
        )
    else:
        bbox_3d = (
            np.array(
                [
                    [scale / 2, +scale / 2, scale / 2],
                    [scale / 2, +scale / 2, -scale / 2],
                    [-scale / 2, +scale / 2, scale / 2],
                    [-scale / 2, +scale / 2, -scale / 2],
                    [+scale / 2, -scale / 2, scale / 2],
                    [+scale / 2, -scale / 2, -scale / 2],
                    [-scale / 2, -scale / 2, scale / 2],
                    [-scale / 2, -scale / 2, -scale / 2],
                ]
            )
            + shift
        )

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack(
        [coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]
    )
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates
def compute_3d_iou(
    RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2, sym_classes
):
    """Computes IoU overlaps between two 3d bboxes.
    bbox_3d_1, bbox_3d_1: [3, 8]
    """

    # flatten masks
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=1)
        bbox_1_min = np.amin(bbox_3d_1, axis=1)
        bbox_2_max = np.amax(bbox_3d_2, axis=1)
        bbox_2_min = np.amin(bbox_3d_2, axis=1)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = (
            np.prod(bbox_1_max - bbox_1_min)
            + np.prod(bbox_2_max - bbox_2_min)
            - intersections
        )
        overlaps = intersections / union
        return overlaps

    if RT_1 is None or RT_2 is None:
        return -1

    if (class_name_1 in sym_classes and class_name_1 == class_name_2) or (
        class_name_1 == "mug"
        and class_name_1 == class_name_2
        and handle_visibility == 0
    ):
        # print('*'*10)


        def y_rotation_matrix(theta):
            return np.array(
                [
                    [np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1],
                ]
            )

        n = 20
        max_iou, max_niou = 0, 0
        for i in range(n):
            rotated_RT_1 = RT_1 @ y_rotation_matrix(2 * math.pi * i / float(n))
            max_iou = max(
                max_iou, asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2)
            )
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    return max_iou


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x
def compute_3d_matches(
    gt_class_ids,
    gt_RTs,
    gt_scales,
    gt_handle_visibility,
    synset_names,
    pred_class_ids,
    pred_scores,
    pred_RTs,
    pred_scales,
    iou_3d_thresholds,
):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)

        pred_class_ids = pred_class_ids[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()
    gt_RTs = gt_RTs.copy()
    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    noverlaps = np.zeros((num_pred, num_gt), dtype=np.float32)

    for i in range(num_pred):
        for j in range(num_gt):
            iou, niou = compute_3d_iou(
                pred_RTs[i],
                gt_RTs[j],
                pred_scales[i],
                gt_scales[j],
                gt_handle_visibility[j],
                synset_names[pred_class_ids[i]],
                synset_names[gt_class_ids[j]],
            )
            overlaps[i, j] = iou
            noverlaps[i,j] = niou

    # Loop through predictions and find matching ground truth boxes

    row_ind, col_ind = linear_sum_assignment(-noverlaps)


    pred_matches, gt_matches = score_from_match(row_ind, col_ind, noverlaps, iou_3d_thresholds)

    return gt_matches, pred_matches, indices

def compute_RT_score(
        RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2, synset_names, class_id, sym_classes
):
    """
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    """

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if synset_names[class_id] in sym_classes:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif synset_names[class_id] == "mug" and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ["phone", "eggbox", "glue"]:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(
            np.arccos((np.trace(R) - 1) / 2), np.arccos((np.trace(R_rot) - 1) / 2)
        )
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def matching(S):
    return linear_sum_assignment(S)

def get_score_matrix_iou(
    gt_class_ids,
    gt_RTs,
    gt_scales,
    gt_handle_visibility,
    synset_names,
    pred_class_ids,
    pred_RTs,
    pred_scales,
    sym_classes
):

    # Trim zero padding
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    S = np.zeros((num_pred, num_gt), dtype=np.float32)

    for i in range(num_pred):
        for j in range(num_gt):
            S[i, j] = compute_3d_iou(
                pred_RTs[i],
                gt_RTs[j],
                pred_scales[i],
                gt_scales[j],
                gt_handle_visibility[j],
                synset_names[pred_class_ids[i]],
                synset_names[gt_class_ids[j]],
                sym_classes
            )
    return S

def get_score_matrix_pose(
    gt_class_ids, gt_RTs, gt_handle_visibility, pred_class_ids, pred_RTs, synset_names

):

    # Trim zero padding
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    S_theta, S_t = np.zeros((2, num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):

            s_theta, s_t = compute_RT_degree_cm_symmetry(
                pred_RTs[i],
                gt_RTs[j],
                gt_class_ids[j],
                gt_handle_visibility[j],
                synset_names,
            )
            S_theta[i, j] = s_theta
            S_t[i, j] = s_t
    return S_theta, S_t

def score_from_match(row_ind, col_ind, score_matrix, thresholds):
    mAPs = []
    num_pred, num_gt = score_matrix[0].shape
    pred_matches = -1 * np.ones([len(n_t) for n_t in thresholds] + [num_pred])
    gt_matches = -1 * np.ones([len(n_t) for n_t in thresholds] + [num_gt])
    for thres, sm in zip(thresholds, score_matrix):
        num_thres = len(thres)

        scores = np.tile(sm[row_ind, col_ind], (num_thres, 1))
        mAP = scores > np.array(thres)[:, None]
        mAPs.append(mAP)

    if len(mAPs) > 1:
        mAP =np.einsum("in,jn -> ijn", mAPs[0], mAPs[1])
    else:
        mAP = mAPs[0]
    pred_matches[..., row_ind] = np.where(mAP, col_ind, -1).astype(np.float32)
    gt_matches[..., col_ind] = np.where(mAP, row_ind, -1).astype(np.float32)

    return gt_matches, pred_matches

def compute_mAP(
        final_results,
        log_dir,
        degree_thresholds,
        shift_thresholds,
        iou_3d_thresholds,
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
        plot_figure=False,
        fig_title="",
        synset_names=None,
        sym_classes=None
):
    if synset_names is None:
        synset_names = ["BG", "bottle", "bowl", "camera", "can", "laptop", "mug"]
    if sym_classes is None:
        sym_classes = ["bottle", "bowl", "can"]

    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100000]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list, "iou_pose_thres should be in iou_thres_list"
    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]
    pose_gt_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)
    ]

    progress = tqdm(final_results, desc="Evaluating", unit="sample")

    for i, result in enumerate(progress):
        gt_class_ids = result["gt_class_ids"].astype(np.int32)
        gt_RTs = np.array(result["gt_RTs"])
        gt_scales = np.array(result["gt_scales"])
        gt_handle_visibility = result["gt_handle_visibility"]

        pred_class_ids = result["pred_class_ids"]
        pred_scales = result["pred_scales"]
        pred_RTs = np.array(result["pred_RTs"])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue
        for cls_id in range(1, num_classes):

            # get gt and predictions in this class
            cls_gt_class_ids = (
                gt_class_ids[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros(0)
            )
            cls_gt_scales = (
                gt_scales[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros((0, 3))
            )
            cls_gt_RTs = (
                gt_RTs[gt_class_ids == cls_id]
                if len(gt_class_ids)
                else np.zeros((0, 4, 4))
            )

            cls_pred_class_ids = (
                pred_class_ids[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros(0)
            )
            cls_pred_RTs = (
                pred_RTs[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros((0, 4, 4))
            )
            cls_pred_scales = (
                pred_scales[pred_class_ids == cls_id]
                if len(pred_class_ids)
                else np.zeros((0, 3))
            )

            # calculate the overlap between each gt instance and pred instance
            if synset_names[cls_id] != "mug":
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = (
                    gt_handle_visibility[gt_class_ids == cls_id]
                    if len(gt_class_ids)
                    else np.ones(0)
                )

            s_1 = np.cbrt(np.linalg.det(cls_gt_RTs[:,:3, :3]))
            s_2 = np.cbrt(np.linalg.det(cls_pred_RTs[:,:3, :3]))
            cls_gt_RTs_norm = cls_gt_RTs.copy()
            cls_pred_RTs_norm = cls_pred_RTs.copy()
            cls_gt_RTs_norm[:,:3, :] /= s_1[:,None,None]
            cls_pred_RTs_norm[:,:3, :] /= s_2[:,None,None]

            noverlaps = get_score_matrix_iou(
                cls_gt_class_ids,
                cls_gt_RTs_norm,
                cls_gt_scales,
                cls_gt_handle_visibility,
                synset_names,
                cls_pred_class_ids,
                cls_pred_RTs_norm,
                cls_pred_scales,
                sym_classes
            )
            noverlaps = np.nan_to_num(noverlaps, nan=0)
            row_ind, col_ind = matching(-noverlaps)

            overlaps = get_score_matrix_iou(
                cls_gt_class_ids,
                cls_gt_RTs,
                cls_gt_scales,
                cls_gt_handle_visibility,
                synset_names,
                cls_pred_class_ids,
                cls_pred_RTs,
                cls_pred_scales,
                sym_classes
            )

            iou_cls_gt_match, iou_cls_pred_match = score_from_match(
                row_ind, col_ind, [overlaps], [iou_thres_list]
            )

            iou_pred_matches_all[cls_id] = np.concatenate(
                (iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1
            )


            iou_gt_matches_all[cls_id] = np.concatenate(
                (iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1
            )

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)
                _, iou_cls_pred_match_norm = score_from_match(
                    row_ind, col_ind, [noverlaps], [iou_thres_list]
                )
                iou_thres_pred_match = iou_cls_pred_match_norm[thres_ind, :]

                cls_pred_class_ids = (
                    cls_pred_class_ids[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros(0)
                )
                cls_pred_RTs = (
                    cls_pred_RTs[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros((0, 4, 4))
                )
                cls_pred_RTs_norm = (
                    cls_pred_RTs_norm[iou_thres_pred_match > -1]
                    if len(iou_thres_pred_match) > 0
                    else np.zeros((0, 4, 4))
                )
                noverlaps = get_score_matrix_iou(
                    cls_gt_class_ids,
                    cls_gt_RTs_norm,
                    cls_gt_scales,
                    cls_gt_handle_visibility,
                    synset_names,
                    cls_pred_class_ids,
                    cls_pred_RTs_norm,
                    cls_pred_scales,
                    sym_classes
                )
                row_ind, col_ind = matching(-noverlaps)

            roverlaps, toverlaps = get_score_matrix_pose(
                cls_gt_class_ids,
                cls_gt_RTs,
                cls_gt_handle_visibility,
                cls_pred_class_ids,
                cls_pred_RTs,
                synset_names,
            )
            try:
                row_ind, col_ind = matching(toverlaps + roverlaps)
            except:
                pass
            pose_cls_gt_match, pose_cls_pred_match = score_from_match(
                row_ind, col_ind, np.stack([-roverlaps, -toverlaps], axis=0),
                ([-d for d in degree_thres_list ], [-st for st in shift_thres_list])
            )

            pose_pred_matches_all[cls_id] = np.concatenate(
                (pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1
            )

            pose_gt_matches_all[cls_id] = np.concatenate(
                (pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1
            )

    iou_dict = {}
    iou_dict["thres_list"] = iou_thres_list
    for cls_id in range(1, num_classes):
        for s, _ in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(
                iou_pred_matches_all[cls_id][s, :],
                iou_gt_matches_all[cls_id][s, :],
            )
        # ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)

    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    # ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    iou_dict["aps"] = iou_3d_aps

    if plot_figure:
        # draw iou 3d AP vs. iou thresholds
        plt.rc("xtick", labelsize=24)  # X 轴刻度标签大小
        plt.rc("ytick", labelsize=24)
        plt.rc("axes", labelsize=26)
        fig_iou = plt.figure(figsize=(30, 10))
        ax_iou = plt.subplot(131)
        plt.ylabel("AP")
        plt.ylim((0, 1))
        plt.xlabel("3D IoU thresholds")
        for cls_id in range(1, num_classes):
            class_name = synset_names[cls_id]
            ax_iou.plot(
                iou_thres_list[:-1],
                iou_3d_aps[cls_id, :-1],
                label=class_name,
                linewidth=4,
            )

        ax_iou.plot(iou_thres_list[:-1], iou_3d_aps[-1, :-1], label="mean", linewidth=4)

    # draw pose AP vs. thresholds
    if use_matches_for_pose:
        prefix = "Pose_Only_"
    else:
        prefix = "Pose_Detection_"

    pose_dict = {}
    pose_dict["degree_thres"] = degree_thres_list
    pose_dict["shift_thres_list"] = shift_thres_list

    for i, _ in enumerate(degree_thres_list):
        for j, _ in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]

                pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(
                    cls_pose_pred_matches_all,
                    cls_pose_gt_matches_all,
                )

            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])

    if plot_figure:
        ax_trans = plt.subplot(132)
        plt.ylim((0, 1))

        plt.xlabel("Rotation/degree")
        for cls_id in range(1, num_classes):
            class_name = synset_names[cls_id]
            ax_trans.plot(
                degree_thres_list[:-1],
                pose_aps[cls_id, :-1, -1],
                label=class_name,
                linewidth=4,
            )

        ax_trans.plot(
            degree_thres_list[:-1], pose_aps[-1, :-1, -1], label="mean", linewidth=4
        )
        pose_dict["aps"] = pose_aps

        ax_rot = plt.subplot(133)
        plt.ylim((0, 1))
        plt.xlabel("Translation/d")
        for cls_id in range(1, num_classes):
            class_name = synset_names[cls_id]
            ax_rot.plot(
                [i / 100 for i in shift_thres_list[:-1]],
                pose_aps[cls_id, -1, :-1],
                label=class_name,
                linewidth=4,
            )

        ax_rot.plot(
            [i / 100 for i in shift_thres_list[:-1]],
            pose_aps[-1, -1, :-1],
            label="mean",
            linewidth=4,
        )
        output_path = os.path.join(
            log_dir,
            fig_title
            +"mAP_{}-{}cm.png".format(shift_thres_list[0], shift_thres_list[-2]),
        )
        ax_rot.legend(fontsize=22)
        fig_iou.savefig(output_path)
        plt.close(fig_iou)



    return iou_3d_aps, pose_aps
def compute_ap_from_matches_scores(pred_match, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)


    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap
def run_evaluation(result_dir, cls_names, sym_classes):
    degree_thres_list = [5, 10, 25, 30, 60, 360]
    shift_thres_list = [5, 10, 1e4]
    iou_thres_list = [0.0,0.1, 0.25, 0.5, 0.75]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, "results_*.pkl"))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, "rb") as f:
            result = cPickle.load(f)
            if "gt_handle_visibility" not in result:
                result["gt_handle_visibility"] = np.ones_like(result["gt_class_ids"])
            else:
                assert len(result["gt_handle_visibility"]) == len(
                    result["gt_class_ids"]
                ), "{} {}".format(
                    result["gt_handle_visibility"], result["gt_class_ids"]
                )
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False

    abs_iou_aps, abs_pose_aps = compute_mAP(
        pred_results,
        result_dir,
        degree_thres_list,
        shift_thres_list,
        iou_thres_list,
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
        plot_figure=True,
        fig_title="Absolute",
        synset_names=cls_names,
        sym_classes=sym_classes
    )
    write_eval_logs(result_dir, abs_iou_aps, abs_pose_aps, degree_thres_list, shift_thres_list, iou_thres_list, metric="Absolute")

    degree_thresholds = [5, 10, 25, 30, 60, 360]
    shift_thresholds = [5, 10, 20, 50]
    iou_3d_thresholds = [0.1, 0.25, 0.5, 0.75]
    # Following LaPose, evaluated on the scale-agonistic metrics
    norm_results = []
    for curr_result in pred_results:
        gt_rts = curr_result["gt_RTs"].copy()
        gt_scale = np.cbrt(np.linalg.det(gt_rts[:, :3, :3]))
        gt_rts[:, :3, :] = gt_rts[:, :3, :] / gt_scale[:, None, None]
        curr_result["gt_RTs"] = gt_rts
        pred_rts = curr_result["pred_RTs"].copy()
        pred_scale = np.cbrt(np.linalg.det(pred_rts[:, :3, :3]))
        pred_rts[:, :3, :] = pred_rts[:, :3, :] / pred_scale[:, None, None]
        curr_result["pred_RTs"] = pred_rts
        norm_results.append(curr_result)

    norm_iou_aps, norm_pose_aps = compute_mAP(
        norm_results,
        result_dir,
        degree_thresholds=degree_thresholds,
        shift_thresholds=shift_thresholds,
        iou_3d_thresholds=iou_3d_thresholds,
        iou_pose_thres=0.1,
        use_matches_for_pose=True,
        plot_figure=True,
        synset_names=cls_names,
        fig_title = "Normalized",
        sym_classes=sym_classes

    )

    write_eval_logs(
        result_dir,
        norm_iou_aps,
        norm_pose_aps,
        degree_thres_list=degree_thresholds,
        shift_thres_list=shift_thresholds,
        iou_thres_list=iou_3d_thresholds,
        metric="Normalized",
    )

    # # plot
    # plot_mAP(
    #     iou_aps,
    #     pose_aps,
    #     result_dir,
    #     iou_thres_list,
    #     degree_thres_list,
    #     shift_thres_list,
    #     metric="Real275",
    #     normalized=False
    # )


def write_eval_logs(
    result_dir,
    iou_aps,
    pose_aps,
    degree_thres_list=None,
    shift_thres_list=None,
    iou_thres_list=None,
    iou_acc=None,
    pose_acc=None,
    metric="",
):

    print(f"Evaluated on Metrics:", metric)

    iou_10_idx = iou_thres_list.index(0.1)
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    degree_25_idx = degree_thres_list.index(25)
    degree_30_idx = degree_thres_list.index(30)
    degree_60_idx = degree_thres_list.index(60)


    # metric
    fw = open("{0}/eval_logs.txt".format(result_dir), "a")
    messages = []
    messages.append(f"{metric} mAP:")

    if metric == "Normalized":
        shift_20_idx = shift_thres_list.index(20)
        shift_50_idx = shift_thres_list.index(50)

        messages.append("3D IoU at 10: {:.1f}".format(iou_aps[-1, iou_10_idx] * 100))
        messages.append("3D IoU at 25: {:.1f}".format(iou_aps[-1, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_aps[-1, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_aps[-1, iou_75_idx] * 100))
        messages.append(
            "5 degree, 20%: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_20_idx] * 100
            )
        )
        messages.append(
            "5 degree, 50%: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_50_idx] * 100
            )
        )
        messages.append(
            "10 degree, 20%: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_20_idx] * 100
            )
        )
        messages.append(
            "10 degree, 50%: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_50_idx] * 100
            )
        )
        messages.append("20%: {:.1f}".format(pose_aps[-1, -1, shift_20_idx] * 100))
        messages.append("50%: {:.1f}".format(pose_aps[-1, -1, shift_50_idx] * 100))
        messages.append(
            "05 degree: {:.1f}".format(pose_aps[-1, degree_05_idx, -1] * 100)
        )
        messages.append(
            "10 degree: {:.1f}".format(pose_aps[-1, degree_10_idx, -1] * 100)
        )
        messages.append(
            "25 degree: {:.1f}".format(pose_aps[-1, degree_25_idx, -1] * 100)
        )
        messages.append(
            "30 degree: {:.1f}".format(pose_aps[-1, degree_30_idx, -1] * 100)
        )
        messages.append(
            "60 degree: {:.1f}".format(pose_aps[-1, degree_60_idx, -1] * 100)
        )
    else:
        shift_05_idx = shift_thres_list.index(5)
        shift_10_idx = shift_thres_list.index(10)

        messages.append("3D IoU at 10: {:.1f}".format(iou_aps[-1, iou_10_idx] * 100))
        messages.append("3D IoU at 25: {:.1f}".format(iou_aps[-1, iou_25_idx] * 100))
        messages.append("3D IoU at 50: {:.1f}".format(iou_aps[-1, iou_50_idx] * 100))
        messages.append("3D IoU at 75: {:.1f}".format(iou_aps[-1, iou_75_idx] * 100))
        messages.append(
            "5 degree, 5cm: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "5 degree, 10cm: {:.1f}".format(
                pose_aps[-1, degree_05_idx, shift_10_idx] * 100
            )
        )
        messages.append(
            "10 degree, 5cm: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_05_idx] * 100
            )
        )
        messages.append(
            "10 degree, 10cm: {:.1f}".format(
                pose_aps[-1, degree_10_idx, shift_10_idx] * 100
            )
        )
        messages.append("5cm: {:.1f}".format(pose_aps[-1, -1, shift_05_idx] * 100))
        messages.append("10cm: {:.1f}".format(pose_aps[-1, -1, shift_10_idx] * 100))
        messages.append(
            "5 degree: {:.1f}".format(pose_aps[-1, degree_05_idx, -1] * 100)
        )
        messages.append(
            "10 degree: {:.1f}".format(pose_aps[-1, degree_10_idx, -1] * 100)
        )
        messages.append(
            "25 degree: {:.1f}".format(pose_aps[-1, degree_25_idx, -1] * 100)
        )
        messages.append(
            "30 degree: {:.1f}".format(pose_aps[-1, degree_30_idx, -1] * 100)
        )
        messages.append(
            "60 degree: {:.1f}".format(pose_aps[-1, degree_60_idx, -1] * 100)
        )

    for msg in messages:
        print(msg)
        fw.write(msg + "\n")
    fw.close()



def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates


def get_bbox(bbox):
    """Compute square image crop window."""
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    # draw pillars in minor darker color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)

    return img


def align_rotation(sRT):
    """Align rotations for symmetric objects.
    Args:
        sRT: 4 x 4
    """
    s = np.cbrt(np.linalg.det(sRT[:3, :3]))
    R = sRT[:3, :3] / s
    T = sRT[:3, 3]

    theta_x = R[0, 0] + R[2, 2]
    theta_y = R[0, 2] - R[2, 0]
    r_norm = math.sqrt(theta_x**2 + theta_y**2)
    s_map = np.array(
        [
            [theta_x / r_norm, 0.0, -theta_y / r_norm],
            [0.0, 1.0, 0.0],
            [theta_y / r_norm, 0.0, theta_x / r_norm],
        ]
    )
    rotation = R @ s_map
    aligned_sRT = np.identity(4, dtype=np.float32)
    aligned_sRT[:3, :3] = s * rotation
    aligned_sRT[:3, 3] = T
    return aligned_sRT