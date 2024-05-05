import tqdm
from torch import nn, optim
import torch
from torchvision.utils import save_image
from data import *
from util.metrics import *
import random

def train(model, train_data_loader, train_dataset, val_data_loader, val_dataset, device, epochs = 100):
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
        class_iou_sums = None
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
                for index in range(6):
                    random_integer = random.randint(0, 319)
                    print(f"Segment Data: {[f'{x:.2f}' for x in segment_image[0, index, ::100, random_integer]]}")
                    print(f"Out Data    : {[f'{x:.2f}' for x in out_image[0, index, ::100, random_integer]]}")

            train_loss = loss_fun(out_image, segment_image)
            train_loss_epoch += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            
            # IoU
            train_iou, train_mean_iou = calculate_iou(out_image, segment_image)
            train_iou_epoch += train_mean_iou.item()
            if class_iou_sums is None:
                class_iou_sums = train_iou
            else:
                class_iou_sums += train_iou

            if view:
                segment_np = segment_image.cpu().detach().numpy()
                out_np = out_image.cpu().detach().numpy()

                # 使用修改后的 one_hot_to_rgb 函数获取 RGB 图像
                segment_rgb = train_dataset.one_hot_to_rgb(segment_np)  # 假设这返回一个 (batch_size, H, W, 3) 的数组
                out_rgb = train_dataset.one_hot_to_rgb(out_np)

                # 将 Numpy 数组转换为 PIL Images
                image_rgb = TF.to_pil_image(image.cpu().squeeze(0))
                segment_rgb = Image.fromarray(segment_rgb.squeeze(0), 'RGB')  # 假设 segment_rgb 只有一个 batch
                out_rgb = Image.fromarray(out_rgb.squeeze(0), 'RGB')

                # 组合图片以保存或展示
                images = [TF.to_tensor(image_rgb), TF.to_tensor(segment_rgb), TF.to_tensor(out_rgb)]
                images = torch.cat(images, dim=2)  # 将三张图片沿宽度方向拼接

                # 保存图片
                save_path = f'./train_image/{i}.gif'
                save_image(images, save_path)
        
        avg_train_loss = train_loss_epoch / len(train_data_loader)
        train_losses.append(avg_train_loss)
        avg_train_iou = train_iou_epoch / len(train_data_loader)
        train_ious.append(avg_train_iou)
        avg_train_class_iou = class_iou_sums / len(train_data_loader)

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
                val_iou, val_mean_iou = calculate_iou(out_image, segment_image)
                val_iou_epoch += val_mean_iou.item()

        avg_val_loss = val_loss_epoch / len(val_data_loader)
        val_losses.append(avg_val_loss)
        avg_val_iou = val_iou_epoch / len(val_data_loader)
        val_ious.append(avg_val_iou)

        # Save model if validation loss or iou has improved
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), './params/best_model_weights.pth')
            print('Model weights saved.')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses}, './params/latest_checkpoint.pth')

        print(f'Epoch:{epoch} - Train Loss: {avg_train_loss} - Train IoU: {avg_train_iou} - Val Loss: {avg_val_loss} - Val IoU: {avg_val_iou}')
        print(f'Average IoU per class: {avg_train_class_iou.tolist()}')

        torch.save(train_losses, 'result/train_losses.tensor')
        torch.save(val_losses, 'result/val_losses.tensor')
        torch.save(train_ious, 'result/train_ious.tensor')
        torch.save(val_ious, 'result/val_ious.tensor')