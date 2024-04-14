import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as distance


def calc_iou(out_image, labels):
    preds = torch.sigmoid(out_image) > 0.5
    preds = preds.long()
    labels = labels.long()
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def calc_iou_multichannel(out_image, labels):
    ious = []
    for channel in range(out_image.shape[1]):
        preds = torch.sigmoid(out_image[:, channel]) > 0.5
        preds = preds.long()
        labels_channel = labels[:, channel].long()
        intersection = (preds & labels_channel).float().sum((1, 2))
        union = (preds | labels_channel).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou.mean())
    return sum(ious) / len(ious)

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


def train(model, train_data_loader, val_data_loader, epochs = 100):
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
            image, segment_image = image.to(device), segment_image.to(device)
            if i%10 == 0:
                view = True
                out_image, middle_x = model(image, view)
            else:
                view = False
                out_image = model(image, view)
            train_loss = loss_fun(out_image, segment_image)
            train_loss_epoch += train_loss.item()
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            # Visualize
            if view:
                _image = image[0]
                _segment_image = segment_image[0]   
                _out_image = out_image[0]
                _out_image_block = Background_Threshold(out_image, threshold=0.95)[0]
                _middle_x_processed = [middle_x[stack_i][0].mean(dim=0).unsqueeze(0).expand(3, -1, -1) if stack_i < len(middle_x)-0 else middle_x[stack_i][0] for stack_i in range(len(middle_x))]
                img = torch.stack([_image] + _middle_x_processed + [_out_image, _out_image_block, _segment_image], dim=0)
                save_image(img, f'{save_path}/{i}.gif')
            
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
                image, segment_image = image.to(device), segment_image.to(device)
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
        # if avg_val_loss < best_val_loss and avg_val_iou > best_val_iou:
        if avg_val_iou > best_val_iou:
            # best_val_loss = avg_val_loss
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), './params/best_model_weights.pth')
            print('Model weights saved.')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'train_loss': train_losses, 'val_loss': val_losses}, './params/latest_checkpoint.pth')

        if i % 1 == 0:
            print(f'Epoch:{epoch} - Train Loss: {avg_train_loss} - Train IoU: {avg_train_iou} - Val Loss: {avg_val_loss} - Val IoU: {avg_val_iou}')

        torch.save(train_losses, 'result/train_losses.tensor')
        torch.save(val_losses, 'result/val_losses.tensor')
        torch.save(train_ious, 'result/train_ious.tensor')
        torch.save(val_ious, 'result/val_ious.tensor')


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    weights_path = 'params'
    weight_path = 'params/best_model_weights.pth'
    data_path = r'data'
    save_path = 'train_image'
    batch_size = 1

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset = MyDataset_tvt(data_path, inputdir='JPEGImages_black', maskdir='SegmentationClass_noYellow', subset="train")
    val_dataset = MyDataset_tvt(data_path, inputdir='JPEGImages_black', maskdir='SegmentationClass_noYellow', subset="val")
    test_dataset = MyDataset_tvt(data_path, inputdir='JPEGImages_black', maskdir='SegmentationClass_noYellow', subset="test")

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet_simple().to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('successful load weight!')
    else:
        print('not successful load weight')

    train(model, train_data_loader, val_data_loader, epochs = 100)