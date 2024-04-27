import tqdm
import torch
from matrix import *

def test(model, test_data_loader, device):
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    loss_fun = torch.nn.CrossEntropyLoss()
    class_iou_sums = None
    with torch.no_grad():
        for image, segment_image in tqdm.tqdm(test_data_loader):
            image, segment_image = image.to(device), segment_image.to(device).float()
            output = model(image, view=False)
            loss = loss_fun(output, segment_image)
            test_loss += loss.item()

            each_iou, iou = calculate_iou(output, segment_image)
            test_iou += iou.item()

            if class_iou_sums is None:
                class_iou_sums = each_iou
            else:
                class_iou_sums += each_iou

    avg_class_iou = class_iou_sums / len(test_data_loader)

    avg_test_loss = test_loss / len(test_data_loader)
    avg_test_iou = test_iou / len(test_data_loader)

    print(f'Test Loss: {avg_test_loss:.4f}, Test IoU: {avg_test_iou:.4f}')
    print(f'Average IoU per class: {avg_class_iou.tolist()}')