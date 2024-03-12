import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image
#from models import UNET
import segmentation_models_pytorch as smp
from utils import load_checkpoint, get_loaders, check_accuracy
import matplotlib.pyplot as plt
from losses import FocalLoss


# Turn off warning related to smp package 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Implicit dimension choice for softmax has been deprecated.*")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 5
#LOAD_MODEL = False
DATA_PATH = "E:\\temp_data_dump\\PanNuke\\data\\png"
SAVE_PATH = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\pred_masks\\Focal Loss"
MODEL_PATH = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_CELoss_f13.pth"
TRAIN_FOLD = ['fold1', 'fold3']
VAL_FOLD = ['fold2']


#MODEL = UNET(in_channels=3, out_channels=1).to(DEVICE)
#load_checkpoint(torch.load(MODEL_PATH), MODEL)
#MODEL.eval()

MODEL = smp.Unet(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=6,                      # model output channels (number of classes in your dataset)
    activation='softmax'
).to(DEVICE)
load_checkpoint(torch.load(MODEL_PATH), MODEL)
MODEL.eval()

train_loader, val_loader = get_loaders(DATA_PATH, TRAIN_FOLD, VAL_FOLD)


#trans = v2.Compose([
#    #v2.Resize((352, 352)),
#    v2.ToImage(), 
#    v2.ToDtype(torch.float32, scale=True)
#    ])                                                                  # for trnasforms.v2


#trans = A.Compose([
#    A.Resize(352, 352),
#    #A.HorizontalFlip(p=0.5),
#    #A.VerticalFlip(p=0.5),
#    A.ToFloat(),
#    ToTensorV2()
#    ])                                                                 # for albumentations

def predict(loader, model = MODEL):
    print("| Saving prediction masks...")
    i = 0
    for batch_idx, (data, target) in enumerate(loader):
        #print(len(loader))
        #print(data.size(0))
        target = target.squeeze(1)
        logits, predictions = torch.max(model(data.to(DEVICE)), dim=1)
        for j in range(data.size(0)):  # Iterate over the batch dimension
            #print(i, j)
            img = data[j].detach().cpu().permute(1, 2, 0).numpy()
            mask = target[j].detach().cpu().numpy()
            pred_mask = predictions[j].detach().cpu().numpy()

            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(mask)
            axes[1].set_title('Mask')
            axes[1].axis('off')
            axes[2].imshow(img)
            axes[2].imshow(mask, alpha=0.35)
            axes[2].set_title('Mask Overlay on original')
            axes[2].axis('off')
            axes[3].imshow(pred_mask)
            axes[3].set_title('Predicted Mask')
            axes[3].axis('off')
            axes[4].imshow(img)
            axes[4].imshow(pred_mask, alpha=0.35)
            axes[4].set_title('Predicted Mask Overlay on original')
            axes[4].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_PATH, f'{i}_Focal_loss.png'))  # Save each image with a unique name
            plt.close()  # Close the figure to avoid memory leaks
            i += 1

#def main():
#    predict(TEST_IMAGES_FOLDER, OUTPUT_MASKS_FOLDER)
#    print("| Prediction and saving complete.")


if __name__ == '__main__':
    #predict(val_loader)
    check_accuracy(val_loader, MODEL, nn.CrossEntropyLoss())
    #print(len(val_loader))