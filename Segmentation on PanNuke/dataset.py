import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PanNuke(Dataset):
    def __init__(self, root_dir, folds = ['fold1', 'fold2'], transform = None):
        self.root_dir = root_dir
        self.folds = folds
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        for fold in self.folds:
            fold_path = os.path.join(root_dir, fold)
            image_path = os.path.join(fold_path, 'images')
            mask_path = os.path.join(fold_path, 'masks')
            image_files = os.listdir(image_path)
            mask_files = os.listdir(mask_path)
            # Ensure the number of images and masks match
            assert len(image_files) == len(mask_files), f"Mismatch in number of images and masks for fold {fold}"

            # Store paths
            self.image_paths.extend([os.path.join(image_path, img) for img in image_files])
            self.mask_paths.extend([os.path.join(mask_path, mask) for mask in mask_files])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        img = Image.open(img_path).convert('RGB')                                       # for transforms.v2
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale               # for transforms.v2


        #img = np.array(Image.open(img_path).convert('RGB'))                             # for albumentations
        #mask = np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale     # for albumentations

        # Apply transformations if provided
        if self.transform is not None:
            '''
            Note: Use albumentations to simultaneously apply **same** augmentations on both image and mask, not possible with trnasforms.v2
            '''
            #transformed = self.transform(image=img, mask=mask)      # for albumentations
            #transformed_image = transformed['image']                # for albumentations
            #transformed_mask = transformed['mask']                  # for albumentations
            #
            #del img, mask       # garbage collection                # for albumentations
            img = self.transform[0](img)                              # for transforms.v2
            mask = self.transform[1](mask)                            # for trnasforms.v2

        #return transformed_image, transformed_mask                  # for albumentations
        return img, mask                                            # for trnasforms.v2