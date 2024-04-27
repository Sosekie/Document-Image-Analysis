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

    # print('intersection: ', intersection)
    # print('union: ', union)
    # print('mean_iou: ', mean_iou)
    # print('iou: ', iou)
    
    # mean_iou = mean_iou.cpu().numpy()
    # iou = iou.cpu().numpy()
    # print(f'mean_iou: {mean_iou[0]:.2f}')
    # print("iou: ", [f"{value:.2f}" for value in iou[0]])
    
    return iou, mean_iou

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