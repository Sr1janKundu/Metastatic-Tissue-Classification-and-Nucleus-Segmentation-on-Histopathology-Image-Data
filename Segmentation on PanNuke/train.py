import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from losses import FocalLoss
#from models import UNET
#import segmentation_models as sm
import segmentation_models_pytorch as smp
from utils import (
    #load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy
)


# Turn off warning related to smp package 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Implicit dimension choice for softmax has been deprecated.*")




# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 25
#LOAD_MODEL = False
DATA_PATH = "E:\\temp_data_dump\\PanNuke\\data\\png"
TRAIN_FOLD = ['fold1', 'fold3']
VAL_FOLD = ['fold2']


#weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
#dice_loss = sm.losses.DiceLoss(class_weights=weights) 
#focal_loss = sm.losses.CategoricalFocalLoss()
#total_loss = dice_loss + (1 * focal_loss)  #

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0.0        # initialization for total loss for current epoch 
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.squeeze(1).to(device=DEVICE)                                             # for transforms.v2
        #targets = targets.to(device=DEVICE).float().unsqueeze(1)                       # for albumentations
        #print(f"\nInput image batch shape: {data.size()}, Input target batch shape: {targets.size()}")             # sanity check
        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            #logits, predictions = torch.max(scores, dim=1)                              # Get predicted class indices
            #print(f"Unique pixel values in target: {torch.unique(targets[0, :, :])}")                              # sanity check
            #print(f"Unique pixel values in preds: {torch.unique(predictions[0, :, :])}")                           # sanity check           
            #print(f'\nData dim: {data.size()}, Target dim: {targets.size()}, Preds dim: {predictions.size()}')     # sanity check
            #print(f'\nData type: {data.dtype}, Target type: {targets.dtype}, Preds type: {predictions.dtype}')     # sanity check
            loss = loss_fn(scores, targets)

        total_loss += loss.item()           # accumulate the loss
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(loader)
    print(f'\nTotal training loss for this epoch: {total_loss}')
    print(f'Average training loss for this epoch: {avg_loss}')


def main():
    '''
    From pytorch hub, with pretrained weights 
    '''
    #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True).to(DEVICE)       
    '''
    Self defined, no pretrained weights
    ''' 
    #model = UNET(in_channels=3, out_channels=1).to(DEVICE)  
    '''
    From segmentation-models-pytorch
    '''
    model = smp.Unet(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,                      # model output channels (number of classes in your dataset)
        activation="softmax"            # Softmax activation on final layer
        ).to(DEVICE)
           
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(DATA_PATH, TRAIN_FOLD, VAL_FOLD)

    #if LOAD_MODEL:
    #    load_checkpoint(torch.load("my_checkpoint.pth"), model)-


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f'| Epoch {epoch+1}:')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            }
        
        save_checkpoint(checkpoint, filename='UNet_ResNet50BackEnd_ImageNetWeights_CELoss_f13.pth')

        # check accuracy every 5 epoch
        if epoch%5 == 0:
            check_accuracy(val_loader, model, loss_fn, device=DEVICE)

if __name__ == "__main__":
    main()