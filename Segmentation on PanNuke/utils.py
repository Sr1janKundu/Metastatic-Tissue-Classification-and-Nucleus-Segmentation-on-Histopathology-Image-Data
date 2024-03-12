import torch
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import PanNuke
from sklearn.metrics import jaccard_score


def get_loaders(path, train_fold = ['fold1', 'fold2'], val_fold = ['fold3']):
    batch_size = 32
    #random_seed = 42
    #torch.manual_seed(random_seed)
    trans_img = v2.Compose([
        #v2.Resize((352, 352)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
        ])
    trans_mask = v2.Compose([
        #v2.Resize((352, 352)),
        v2.ToImage(), 
        v2.ToDtype(torch.long, scale=False)
        ])                                                                               # for trnasforms.v2
    
    #trans = A.Compose([
    #    A.Resize(352, 352),    
    #    A.HorizontalFlip(p=0.5),
    #    A.VerticalFlip(p=0.5),
    #    A.ToFloat(),
    #    ToTensorV2()
    #    ])                                                                              # for albumentations
    
    train_ds, val_ds = PanNuke(root_dir=path, folds = train_fold, transform=[trans_img, trans_mask]), PanNuke(root_dir=path, folds = val_fold, transform=[trans_img, trans_mask])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch_size)
    
    return train_dl, valid_dl


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, criterion, device = 'cuda'):
    model.eval()
    total_loss = 0.0
    tot_num_corr = 0
    tot_px = 0
    scores_dict = {f"Class {class_label}": 0.0 for class_label in range(6)}
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            _, preds = torch.max(scores, dim = 1)
            assert torch.numel(preds) == torch.numel(y), f"Mismatch in number of elements in predicted and actual labels"
            loss = criterion(scores, y.squeeze(1))
            total_loss += loss.item()
            y_flattened = torch.flatten(y).detach().cpu().numpy()
            y_preds_flattened = torch.flatten(preds).detach().cpu().numpy()
            tot_num_corr += (y_flattened == y_preds_flattened).sum()
            tot_px += torch.numel(y)
            score = jaccard_score(y_pred=y_preds_flattened, y_true=y_flattened, average= None)
            for class_label, score in enumerate(score):
                #print(f"Jaccard score for class {class_label}: {score}")
                scores_dict[f"Class {class_label}"] += score

            # garbage colloect
            del x, y, scores, preds, loss, y_flattened, y_preds_flattened, score

    print(f"\nPredicted {tot_num_corr} pixels correct out of {tot_px} pixels, with accuracy: {tot_num_corr/tot_px*100:.2f}%")
    print(f'\nTotal validation loss: {total_loss}, Average validation loss: {total_loss/len(loader):.2f}')
    class_scores = [score / len(loader) for score in scores_dict.values()]
    formatted_class_scores = [f"{score:.3f}" for score in class_scores]
    print(f"Average Jaccard scores for the 6 classes - Neoplastic, Limfo, Connective, Dead, Epithelia, Void are: {formatted_class_scores} respectively.")
    model.train()


#def check_accuracy(loader, model, criteria, device="cuda"):
#    num_correct = 0
#    num_pixels = 0
#    dice_score = 0
#    total_loss = 0
#    intersection = 0
#    union = 0
#    IoU = 0
#    model.eval()
#
#    with torch.no_grad():
#        for x, y in loader:
#            x = x.to(device)
#            y = y.to(device)                                            # for transforms.v2
#            #y = y.to(device).float().unsqueeze(1)                       # for albumentations
#            #preds = torch.sigmoid(model(x))
#            #preds = (preds > 0.5).float()
#            scores = model(x)
#            loss = criteria(scores, y.squeeze(1))
#            total_loss += loss
#            logits, preds = torch.max(scores, dim = 1)
#            num_correct += 45045342
#            num_pixels += torch.numel(preds)
#            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
#            intersection = (preds * y).sum()
#            union = (preds + y).sum() - intersection
#            IoU += (intersection + 1e-8) / (union + 1e-8)
#
#        avg_loss = total_loss / len(loader)
#    print(f'\nTotal validation loss: {total_loss}, Average validation loss: {avg_loss}')
#    print(f"\nPredicted {num_correct} pixels correct out of {num_pixels} pixels, with accuracy: {num_correct/num_pixels*100:.2f}%")
#    print(f"Dice score: {dice_score/len(loader)}")
#    print(f"IoU: {IoU / len(loader)}")
#    model.train()