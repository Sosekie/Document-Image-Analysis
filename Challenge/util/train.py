import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import tqdm
import numpy as np
from util.metrics import calculate_iou, calculate_f1_score
import matplotlib.pyplot as plt


def one_hot_to_mask(one_hot_mask, save_path='train_image', file_name='output_image.png'):
    colors = [[0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 0, 0]]

    thresholded_mask = (one_hot_mask >= 0.5).float()
    permuted_thresholded_mask = thresholded_mask.permute(1, 2, 0)
    one_hot_mask_numpy = permuted_thresholded_mask.cpu().numpy()
    # print(one_hot_mask_numpy.shape)

    mask = np.zeros((*one_hot_mask_numpy.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask[one_hot_mask_numpy[:, :, i] == 1] = color

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    mask = Image.fromarray(mask)
    mask.save(os.path.join(save_path, file_name))


def train(model, train_data_loader, val_data_loader, device, epochs=100):
    opt = optim.Adam(model.parameters())
    loss_fun = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    train_f1s = []
    val_f1s = []
    best_val_iou = 0.0
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        train_iou_epoch = 0.0
        train_class_iou_sums = None
        train_f1_epoch = 0.0
        train_class_f1_sums = None
        
        print('Train epoch')
        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data_loader)):
            image, segment_image = image.to(device), segment_image.to(device).float()
            out_image = model(image)

            # if i == 0:
            #     for index in range(6):
            #         random_integer = random.randint(0, 319)
            #         print(f"Segment Data: {[f'{x:.2f}' for x in segment_image[0, index, ::100, random_integer]]}")
            #         print(f"Out Data    : {[f'{x:.2f}' for x in out_image[0, index, ::100, random_integer]]}")

            train_loss = loss_fun(out_image, segment_image)
            train_loss_epoch += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            train_iou, train_mean_iou = calculate_iou(out_image, segment_image)
            train_iou_epoch += train_mean_iou.item()
            if train_class_iou_sums is None:
                train_class_iou_sums = train_iou
            else:
                train_class_iou_sums += train_iou

            train_f1, train_mean_f1 = calculate_f1_score(out_image, segment_image)
            train_f1_epoch += train_mean_f1.item()
            if train_class_f1_sums is None:
                train_class_f1_sums = train_f1
            else:
                train_class_f1_sums += train_f1

            if i % 10 == 0:
                one_hot_to_mask(segment_image[0], file_name='segment_rgb_'+str(i)+'.png')
                one_hot_to_mask(out_image[0], file_name='out_rgb_'+str(i)+'.png')

        avg_train_loss = train_loss_epoch / len(train_data_loader)
        train_losses.append(avg_train_loss)
        avg_train_iou = train_iou_epoch / len(train_data_loader)
        train_ious.append(avg_train_iou)
        avg_train_class_iou = train_class_iou_sums / len(train_data_loader)
        avg_train_f1 = train_f1_epoch / len(train_data_loader)
        train_f1s.append(avg_train_f1)
        avg_train_class_f1 = train_class_f1_sums / len(train_data_loader)

        model.eval()
        val_loss_epoch = 0.0
        val_iou_epoch = 0.0
        val_class_iou_sums = None
        val_f1_epoch = 0.0
        val_class_f1_sums = None
        print('Val epoch')
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(tqdm.tqdm(val_data_loader)):
                image, segment_image = image.to(device), segment_image.to(device).float()
                out_image = model(image)
                val_loss = loss_fun(out_image, segment_image)
                val_loss_epoch += val_loss.item()
                
                val_iou, val_mean_iou = calculate_iou(out_image, segment_image)
                val_iou_epoch += val_mean_iou.item()
                if val_class_iou_sums is None:
                    val_class_iou_sums = val_iou
                else:
                    val_class_iou_sums += val_iou

                val_f1, val_mean_f1 = calculate_f1_score(out_image, segment_image)
                val_f1_epoch += val_mean_f1.item()
                if val_class_f1_sums is None:
                    val_class_f1_sums = val_f1
                else:
                    val_class_f1_sums += val_f1

        avg_val_loss = val_loss_epoch / len(val_data_loader)
        val_losses.append(avg_val_loss)
        avg_val_iou = val_iou_epoch / len(val_data_loader)
        val_ious.append(avg_val_iou)
        avg_val_class_iou = val_class_iou_sums / len(val_data_loader)
        avg_val_f1 = val_f1_epoch / len(val_data_loader)
        val_f1s.append(avg_val_f1)
        avg_val_class_f1 = val_class_f1_sums / len(val_data_loader)

        if avg_val_iou > best_val_iou and avg_val_f1 > best_val_f1:
            best_val_iou = avg_val_iou
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), './params/best_model_weights.pth')
            print('Model weights saved.')

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses}, './params/latest_checkpoint.pth')

        print(f'Epoch:{epoch} - Train Loss: {avg_train_loss} - Train IoU: {avg_train_iou} - Train F1: {avg_train_f1} - Val Loss: {avg_val_loss} - Val IoU: {avg_val_iou} - Val F1: {avg_val_f1}')
        print(f'Average Train IoU per class: {avg_train_class_iou.tolist()}')
        print(f'Average Train F1 per class : {avg_train_class_f1.tolist()}')
        print(f'Average Val IoU per class  : {avg_val_class_iou.tolist()}')
        print(f'Average Val F1 per class   : {avg_val_class_f1.tolist()}')

        torch.save(train_losses, 'result/train_losses.tensor')
        torch.save(val_losses, 'result/val_losses.tensor')
        torch.save(train_ious, 'result/train_ious.tensor')
        torch.save(val_ious, 'result/val_ious.tensor')
        torch.save(train_f1s, 'result/train_f1s.tensor')
        torch.save(val_f1s, 'result/val_f1s.tensor')