import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.transforms import v2
#from torchmetrics.classification import BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy, AUROC
from tqdm.auto import tqdm
from dataset_dataloader import get_loader
from utils import load_checkpoint, save_checkpoint, evaluate


# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
#RANDOM_SEED = 42
DATASET_PATH = 'D:\\dissertation ideas\\PCam'
MODEL_SAVE_PATH = "D:\\dissertation ideas\\PCam\\Implementation\\model_resnet34.pth"
LEARNING_RATE = 0.01
CLASSES = 1
EPOCH = 5


train_dl, val_dl, test_dl = get_loader(batch=BATCH_SIZE)


## From torchvision with pre-trained ImageNet weights
model_resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
model_resnet.to(DEVICE)

model_densnet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
#model_densnet121.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)       # 1 as greyscale
num_ftrs = model_densnet121.classifier.in_features
model_densnet121.classifier = nn.Linear(in_features=num_ftrs, out_features=CLASSES)
model_densnet121.to(DEVICE)


#metric = BinaryF1Score(threshold=0.5, device=DEVICE)
#prec = BinaryPrecision(threshold=0.5, device=DEVICE)
#recall = BinaryRecall(threshold=0.5, device=DEVICE)
#acc = BinaryAccuracy(threshold=0.5, device=DEVICE)
#auc = AUROC(task="binary").cuda()

def train(epochs, model):
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(epochs):
        print(f"\n | Epoch: {epoch+1}")
        total_loss = 0
        loop = tqdm(train_dl)
        for _, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            #with torch.set_grad_enabled(True):
            outputs = model(inputs).squeeze(1)
            loss = loss_func(outputs, labels.float())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss/len(train_dl)
        print(f"| Epoch {epoch+1}/{epochs} total training loss: {total_loss}, average training loss: {avg_loss}.")
        print("On Validation Data:")
        model.eval()
        with torch.inference_mode():
            evaluate(val_dl, model)
        print('Saving model...')
        checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                }
        save_checkpoint(checkpoint, MODEL_SAVE_PATH)
        print(f'Model saved at {MODEL_SAVE_PATH}')


if __name__ == "__main__":
    train(epochs=EPOCH, model=model_resnet)
    print("\n\nOn Test Data:")
    load_checkpoint(torch.load(MODEL_SAVE_PATH), model_resnet)
    model_resnet.eval()
    evaluate(test_dl, model_resnet)