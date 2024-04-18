import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.transforms import v2
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, AUROC
#from torcheval.metrics import BinaryF1Score, BinaryRecall, BinaryPrecision, BinaryAccuracy
from tqdm.auto import tqdm
from dataset_dataloader import get_loader
from utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
#RANDOM_SEED = 42
DATASET_PATH = 'D:\\dissertation ideas\\PCam'
MODEL_SAVE_PATH = "D:\\dissertation ideas\\PCam\\Implementation\\model_resnet50.pth"
LEARNING_RATE = 0.01
CLASSES = 1
EPOCH = 5


train_dl, val_dl, test_dl = get_loader(batch=BATCH_SIZE)

model_resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(in_features=num_ftrs, out_features=1)
model_resnet.to(DEVICE)


model_densnet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
#model_densnet121.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)       # 1 as greyscale
num_ftrs = model_densnet121.classifier.in_features
model_densnet121.classifier = nn.Linear(in_features=num_ftrs, out_features=CLASSES)
model_densnet121.to(DEVICE)


def test(loader, model):
    metric = BinaryF1Score().cuda()
    prec = BinaryPrecision().cuda()
    recall = BinaryRecall().cuda()
    acc = BinaryAccuracy().cuda()
    auc = AUROC(task="binary").cuda()
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    loss = 0.0
    for data, targets in tqdm(loader):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        #preds = outputs.sigmoid()
        preds = torch.round(outputs.sigmoid().squeeze(1))
        loss += loss_func(outputs.squeeze(1), targets.float()).item()
        #print(loss.item())
        #print(data.size(), targets.size(), outputs.size())
        #print(targets)
        #print(preds)
        #print((targets == preds).sum()/targets.size(0))
        metric.update(preds, targets)
        prec.update(preds, targets)
        recall.update(preds, targets)
        acc.update(preds, targets)
        auc.update(outputs.squeeze(1), targets)
    avg_loss = loss/len(loader)
    print(f"Loss: {avg_loss}")
    print(f"AUC: {auc.compute():.3f}, Accuracy: {acc.compute():.3f}, precision: {prec.compute():.3f}, recall: {recall.compute():.3f}, F1Score: {metric.compute():.3f}")
    model.train()
        


if __name__ == '__main__':
    #test(val_dl, model_resnet)
    load_checkpoint(torch.load(MODEL_SAVE_PATH), model_resnet)
    model_resnet.eval()
    test(test_dl, model_resnet)