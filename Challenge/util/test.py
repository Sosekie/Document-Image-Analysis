import torch
import tqdm
from util.metrics import calculate_iou, calculate_f1_score

def test(model, test_data_loader, device):
    model.eval()
    test_ious = []
    test_f1s = []
    test_class_iou_sums = None
    test_class_f1_sums = None
    
    print('Test phase')
    with torch.no_grad():
        for image, segment_image in tqdm.tqdm(test_data_loader):
            image, segment_image = image.to(device), segment_image.to(device).float()
            out_image = model(image)
            
            test_iou, test_mean_iou = calculate_iou(out_image, segment_image)
            test_f1, test_mean_f1 = calculate_f1_score(out_image, segment_image)
            
            if test_class_iou_sums is None:
                test_class_iou_sums = test_iou
            else:
                test_class_iou_sums += test_iou

            if test_class_f1_sums is None:
                test_class_f1_sums = test_f1
            else:
                test_class_f1_sums += test_f1


    avg_test_iou = test_class_iou_sums / len(test_data_loader)
    avg_test_f1 = test_class_f1_sums / len(test_data_loader)

    print(f'Total Test IoU: {avg_test_iou.mean().item()}')
    print(f'Total Test F1: {avg_test_f1.mean().item()}')
    print(f'Average Test IoU per class: {avg_test_iou.tolist()}')
    print(f'Average Test F1 per class: {avg_test_f1.tolist()}')

    return avg_test_iou, avg_test_f1
