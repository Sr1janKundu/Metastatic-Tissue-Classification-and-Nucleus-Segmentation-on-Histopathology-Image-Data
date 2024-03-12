import torch
import torchvision
from torchvision.transforms import v2


DATA_PATH = 'D:\\dissertation ideas\\PCam'

TRANS = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True)
])

def get_loader(root_dir = DATA_PATH, img_trans = TRANS, target_trans = None, batch = 32, seed = None, download = False):
    
    if seed is not None:
        torch.manual_seed(seed)

    train_ds = torchvision.datasets.PCAM(root_dir, 
                                    split = 'train',
                                    transform = img_trans,
                                    target_transform = None,
                                    download = False)
    val_ds = torchvision.datasets.PCAM(root_dir, 
                                    split = 'val',
                                    transform = img_trans,
                                    target_transform = None,
                                    download = False)
    test_ds = torchvision.datasets.PCAM(root_dir, 
                                    split = 'test',
                                    transform = img_trans,
                                    target_transform = None,
                                    download = False)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size = batch, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size = batch, shuffle=False)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size = batch, shuffle=False)

    return train_dl, val_dl, test_dl
    