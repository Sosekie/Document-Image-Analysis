import torch
from scipy.ndimage import distance_transform_edt as distance
import numpy as np

def calculate_iou(out_image, segment_image, threshold=0.5):
    out_image_binary = (out_image > threshold).float()
    segment_image_binary = (segment_image > threshold).float()
    
    intersection = torch.logical_and(out_image_binary, segment_image_binary).float().sum(dim=(2, 3)) + 1
    union = torch.logical_or(out_image_binary, segment_image_binary).float().sum(dim=(2, 3)) + 1
    iou = intersection / union
    mean_iou = iou.mean(dim=1)

    return iou, mean_iou

def calculate_f1_score(out_image, segment_image, threshold=0.5):
    out_image_binary = (out_image > threshold).float()
    segment_image_binary = (segment_image > threshold).float()
    tp = torch.logical_and(out_image_binary, segment_image_binary).float().sum(dim=(2, 3))
    fp = torch.logical_and(out_image_binary, torch.logical_not(segment_image_binary)).float().sum(dim=(2, 3))
    fn = torch.logical_and(torch.logical_not(out_image_binary), segment_image_binary).float().sum(dim=(2, 3))
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    mean_f1 = f1_score.mean(dim=1)
    # print('precision - ', precision)
    # print('recall - ', recall)
    # print('f1_score - ', f1_score)
    # print('mean_f1 - ', mean_f1)

    return f1_score, mean_f1

def calculate_fn_fp(out_image, segment_image, threshold=0.5):
    out_image_binary = (out_image > threshold).float()
    segment_image_binary = (segment_image > threshold).float()
    fp = torch.logical_and(out_image_binary, torch.logical_not(segment_image_binary)).float().sum(dim=(2, 3))
    fn = torch.logical_and(torch.logical_not(out_image_binary), segment_image_binary).float().sum(dim=(2, 3))
    fn_fp_sum = fn + fp

    return fn_fp_sum

def binary_distance_transform(batch):
    batch_transformed = np.zeros_like(batch)
    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            pos_transform = distance(batch[i, j])
            neg_transform = distance(1 - batch[i, j])
            batch_transformed[i, j] = np.where(batch[i, j] > 0, pos_transform, neg_transform)
    return batch_transformed

def boundary_loss(preds, targets):
    preds = torch.sigmoid(preds)
    preds_bin = (preds > 0.5).float()
    targets_bin = (targets > 0.5).float()
    preds_np = preds_bin.cpu().numpy()
    targets_np = targets_bin.cpu().numpy()
    preds_dt = torch.from_numpy(binary_distance_transform(preds_np)).float().to(preds.device)
    targets_dt = torch.from_numpy(binary_distance_transform(targets_np)).float().to(targets.device)
    loss = F.mse_loss(preds_dt, targets_dt, reduction='none')
    loss = loss.mean(dim=(1, 2, 3)).mean()
    return loss