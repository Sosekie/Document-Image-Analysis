import torch
from scipy.ndimage import distance_transform_edt as distance
import numpy as np

def calc_iou(out_image, labels):
    preds = torch.sigmoid(out_image) > 0.5
    preds = preds.long()
    labels = labels.long()
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def calc_iou(out_image, labels):
    preds = (out_image*5).long()
    labels = (labels*5).long()
    # print('preds: ', preds[0][240][150:170])
    # print('labels: ', labels[0][240][150:170])
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# def calc_iou(out_image, labels, num_classes=6):
#     print('out_image: ', out_image[0][0][0][:20])
#     print('labels: ', labels[0][0][:20])
#     # 将标签转换为独热编码格式
#     batch_size, height, width = labels.size()
#     one_hot_labels = torch.zeros(batch_size, num_classes, height, width, device=labels.device)
#     one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

#     # 对预测应用 sigmoid 函数并设置阈值
#     preds = torch.sigmoid(out_image) > 0.5
#     preds = preds.long()  # 确保是整数类型，适用于位运算

#     one_hot_labels = one_hot_labels.long()  # 确保是整数类型，适用于位运算

#     # 计算交集和并集
#     intersection = (preds & one_hot_labels).sum(dim=(2, 3))  # 使用整数类型进行按位与运算
#     union = (preds | one_hot_labels).sum(dim=(2, 3))  # 使用整数类型进行按位或运算

#     # 计算 IoU
#     iou = (intersection + 1e-6) / (union + 1e-6)
#     return iou.mean()

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