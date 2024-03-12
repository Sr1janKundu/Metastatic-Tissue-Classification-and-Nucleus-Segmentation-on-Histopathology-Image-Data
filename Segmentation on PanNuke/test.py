import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils import load_checkpoint, get_loaders
from losses import FocalLoss
from sklearn.metrics import jaccard_score
from tqdm import tqdm

# Turn off warning related to smp package 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Implicit dimension choice for softmax has been deprecated.*")



DATA_PATH = "E:\\temp_data_dump\\PanNuke\\data\\png"
MODEL_PATH = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_FocalLoss_f12.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL = smp.Unet(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=6,                      # model output channels (number of classes in your dataset)
    activation='softmax'
).to(DEVICE)
load_checkpoint(torch.load(MODEL_PATH), MODEL)
MODEL.eval()

def test(loader, model, criterion):
    #print(len(loader))
    total_loss = 0.0
    num_corr = 0
    total_px = 0
    scores_dict = {f"Class {class_label}": 0.0 for class_label in range(6)}
    #intersection = 0
    #union = 0
    #dice = 0
    #iou = 0
    i = 0
    with torch.inference_mode():
        for x, y in tqdm(loader):
            #i+=1
            print(i)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            scores = model(x)
            #print(scores)
            logits, preds = torch.max(scores, dim=1)
            loss = criterion(scores, y.squeeze(1))
            total_loss+=loss.item()
            y = y.to('cpu')
            preds = preds.to('cpu')
            num_corr = (torch.flatten(preds).numpy() == torch.flatten(y).numpy()).sum()
            total_px = torch.numel(y)
            #print(f"\nPredicted {num_corr} pixels correct out of {total_px} pixels, with accuracy: {num_corr/total_px*100:.2f}%")
            score = jaccard_score(y_pred=torch.flatten(preds), y_true=torch.flatten(y), average= None)
            for class_label, score in enumerate(score):
                #print(f"Jaccard score for class {class_label}: {score}")
                scores_dict[f"Class {class_label}"] += score
            #print(f"Jaccard similarity coefficient score: {score}")
            #print(preds.dtype, y.dtype)
            assert torch.numel(preds) == torch.numel(y), f"Mismatch in no. of elements in predicted and actual labels"
            #print(torch.unique(preds), torch.unique(y))
            #print(num_corr)
            #print(torch.numel(preds), torch.numel(y))
            #print(x.size(), y.size(), scores.size(), logits.size(), preds.size(), loss.item())
            #print(preds)
            #print(logits)
            #break
        
        print(f"Total loss: {total_loss}")
        for class_label, score in scores_dict.items():
            print(f"Average Jaccard score for {class_label}: {score/len(loader):.3f}")
if __name__ == '__main__':
    _, val_loader = get_loaders(DATA_PATH)
    #loss_fn = FocalLoss()
    loss_fn = nn.CrossEntropyLoss()
    test(val_loader, MODEL, loss_fn)