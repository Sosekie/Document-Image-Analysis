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
from matrix import calculate_iou  # Import a function that calculates IoU
import matplotlib.pyplot as plt

def restore_mask(one_hot_mask):
    # Convert one-hot encoded mask back to RGB image
    colors = np.array([(0,255,255), (255,0,255), (0,255,0), (255,0,0), (0,0,255), (0,0,0)], dtype=np.uint8)
    rgb_mask = np.dot(one_hot_mask.astype(np.float32), colors)
    return Image.fromarray(rgb_mask.astype(np.uint8))

def train(model, train_data_loader, val_data_loader, device, epochs=100):
    opt = optim.Adam(model.parameters())
    loss_fun = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    best_val_iou = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        train_iou_epoch = 0.0
        class_iou_sums = None
        
        print('Train epoch')
        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data_loader)):
            image, segment_image = image.to(device), segment_image.to(device).float()
            out_image = model(image)

            if i == 0:
                for index in range(6):
                    random_integer = random.randint(0, 319)
                    print(f"Segment Data: {[f'{x:.2f}' for x in segment_image[0, index, ::100, random_integer]]}")
                    print(f"Out Data    : {[f'{x:.2f}' for x in out_image[0, index, ::100, random_integer]]}")

            train_loss = loss_fun(out_image, segment_image)
            train_loss_epoch += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            train_iou, train_mean_iou = calculate_iou(out_image, segment_image)
            train_iou_epoch += train_mean_iou.item()
            if class_iou_sums is None:
                class_iou_sums = train_iou
            else:
                class_iou_sums += train_iou

            if i % 10 == 0:
                segment_rgb = restore_mask(segment_image.cpu().detach().numpy())
                out_rgb = restore_mask(out_image.cpu().detach().numpy())

                image_rgb = TF.to_pil_image(image.cpu().squeeze(0))
                segment_rgb = Image.fromarray(segment_rgb.squeeze(0), 'RGB')
                out_rgb = Image.fromarray(out_rgb.squeeze(0), 'RGB')

                images = [TF.to_tensor(image_rgb), TF.to_tensor(segment_rgb), TF.to_tensor(out_rgb)]
                images = torch.cat(images, dim=2)
                
                save_path = f'./train_image/{i}.png'
                save_image(images, save_path)

        avg_train_loss = train_loss_epoch / len(train_data_loader)
        train_losses.append(avg_train_loss)
        avg_train_iou = train_iou_epoch / len(train_data_loader)
        train_ious.append(avg_train_iou)
        avg_train_class_iou = class_iou_sums / len(train_data_loader)

        model.eval()
        val_loss_epoch = 0.0
        val_iou_epoch = 0.0
        print('Val epoch')
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(tqdm.tqdm(val_data_loader)):
                image, segment_image = image.to(device), segment_image.to(device).float()
                out_image = model(image)
                val_loss = loss_fun(out_image, segment_image)
                val_loss_epoch += val_loss.item()
                
                val_iou, val_mean_iou = calculate_iou(out_image, segment_image)
                val_iou_epoch += val_mean_iou.item()

        avg_val_loss = val_loss_epoch / len(val_data_loader)
        val_losses.append(avg_val_loss)
        avg_val_iou = val_iou_epoch / len(val_data_loader)
        val_ious.append(avg_val_iou)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), './params/best_model_weights.pth')
            print('Model weights saved.')

        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses}, './params/latest_checkpoint.pth')

        print(f'Epoch:{epoch} - Train Loss: {avg_train_loss} - Train IoU: {avg_train_iou} - Val Loss: {avg_val_loss} - Val IoU: {avg_val_iou}')
        print(f'Average IoU per class: {avg_train_class_iou.tolist()}')


def test_one_hot_to_rgb():
    # Create a dummy one-hot encoded mask with known values
    one_hot = np.zeros((1, 480, 320, 6), dtype=np.float32)
    one_hot[0, :240, :, 0] = 1  # Upper half to the first color
    one_hot[0, 240:, :, 1] = 1  # Lower half to the second color
    
    rgb = restore_mask(one_hot)
    plt.imshow(rgb[0])
    plt.title("Test One-hot to RGB Conversion")
    plt.show()

test_one_hot_to_rgb()