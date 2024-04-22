import tqdm
from torch import nn, optim
import torch
from torchvision.utils import save_image

from data import *
from matrix import *

def middle_visualize(image, segment_image, out_image, middle_x, i):
    _image = image[0]
    _segment_image = segment_image[0]
    _out_image = out_image[0]
    _out_image_block = Background_Threshold(out_image, threshold=0.95)[0]
    _middle_x_processed = [middle_x[stack_i][0].mean(dim=0).unsqueeze(0).expand(3, -1, -1) if stack_i < len(middle_x)-0 else middle_x[stack_i][0] for stack_i in range(len(middle_x))]
    img = torch.stack([_image] + _middle_x_processed + [_out_image, _out_image_block, _segment_image], dim=0)
    save_image(img, f'{save_path}/{i}.gif')

def train(model, train_data_loader, val_data_loader, device, epochs = 100):
    opt = optim.Adam(model.parameters())
    loss_fun = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    best_val_loss = float('inf')
    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        train_iou_epoch = 0.0
        print('Train epoch')
        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data_loader)):
            image, segment_image = image.to(device), segment_image.to(device).float()

            if i%10 == 0:
                view = True
                out_image, middle_x = model(image, view)
            else:
                view = False
                out_image = model(image, view)
            if i == 0:
                print(segment_image[0, 0, 235:245, 160], out_image[0, 0, 235:245, 160])
                print(segment_image[0, 1, 235:245, 160], out_image[0, 1, 235:245, 160])
                print(segment_image[0, 2, 235:245, 160], out_image[0, 2, 235:245, 160])
                print(segment_image[0, 3, 235:245, 160], out_image[0, 3, 235:245, 160])
                print(segment_image[0, 4, 235:245, 160], out_image[0, 4, 235:245, 160])
                print(segment_image[0, 5, 235:245, 160], out_image[0, 5, 235:245, 160])

            train_loss = loss_fun(out_image, segment_image)
            train_loss_epoch += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            # IoU
            train_iou = calc_iou(out_image, segment_image)
            train_iou_epoch += train_iou.item()

        avg_train_loss = train_loss_epoch / len(train_data_loader)
        train_losses.append(avg_train_loss)
        avg_train_iou = train_iou_epoch / len(train_data_loader)
        train_ious.append(avg_train_iou)

        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        val_iou_epoch = 0.0
        print('Val epoch')
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(tqdm.tqdm(val_data_loader)):
                image, segment_image = image.to(device), segment_image.to(device).float()
                out_image = model(image, view=False)
                val_loss = loss_fun(out_image, segment_image)
                val_loss_epoch += val_loss.item()
                # IoU
                val_iou = calc_iou(out_image, segment_image)
                val_iou_epoch += val_iou.item()

        avg_val_loss = val_loss_epoch / len(val_data_loader)
        val_losses.append(avg_val_loss)
        avg_val_iou = val_iou_epoch / len(val_data_loader)
        val_ious.append(avg_val_iou)

        # Save model if validation loss or iou has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './params/best_model_weights.pth')
            print('Model weights saved.')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses}, './params/latest_checkpoint.pth')

        if i % 1 == 0:
            print(f'Epoch:{epoch} - Train Loss: {avg_train_loss} - Train IoU: {avg_train_iou} - Val Loss: {avg_val_loss} - Val IoU: {avg_val_iou}')

        torch.save(train_losses, 'result/train_losses.tensor')
        torch.save(val_losses, 'result/val_losses.tensor')
        torch.save(train_ious, 'result/train_ious.tensor')
        torch.save(val_ious, 'result/val_ious.tensor')