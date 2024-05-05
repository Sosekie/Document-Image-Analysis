import torch
import matplotlib.pyplot as plt

def visualize(train_losses, val_losses, train_ious, val_ious, train_f1s, val_f1s, filename = 'visualize'):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_ious, label='Training IoU')
    plt.plot(val_ious, label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.title('Training and Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == '__main__':

    train_losses = torch.load('result/train_losses.tensor')
    val_losses = torch.load('result/val_losses.tensor')
    train_ious = torch.load('result/train_ious.tensor')
    val_ious = torch.load('result/val_ious.tensor')
    train_f1s = torch.load('result/train_f1s.tensor')
    val_f1s = torch.load('result/val_f1s.tensor')

    visualize(train_losses, val_losses, train_ious, val_ious, train_f1s, val_f1s)