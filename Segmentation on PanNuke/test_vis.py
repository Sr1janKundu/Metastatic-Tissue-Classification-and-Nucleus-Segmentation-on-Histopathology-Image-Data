import os
import random
import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import v2
from utils import *
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


DATA_PATH = "E:\\temp_data_dump\\PanNuke\\data\\png"
SAVE_PATH = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\pred_masks\\Focal Loss"
MODEL_PATH = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_CELoss_f13.pth"
TEST_IMAGES_FOLDER = "E:\\temp_data_dump\\PanNuke\\data\\png\\fold3"
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

TRAIN_FOLD = ['fold1', 'fold2']
VAL_FOLD = ['fold3']
_, val_loader = get_loaders(DATA_PATH, TRAIN_FOLD, VAL_FOLD)


#def test(model, image_path):
#    img_trans = v2.Compose([
#        v2.ToImage(),
#        v2.ToDtype(torch.float32, scale=True)
#    ])
#    img = img_trans(Image.open(image_path).convert('RGB'))
#    img = img.unsqueeze(0)
#    print(img.size())
#    _, pred = torch.max(model(img.to(DEVICE)), dim = 1)
#    pred = pred.squeeze(0)
#    print(pred.size())
#
#
#if __name__ == '__main__':
#    test(MODEL, os.path.join(DATA_PATH, 'fold1', 'images', '0.png'))

def perform_inference(model, image_path):
    img_trans = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    img = img_trans(Image.open(image_path).convert('RGB'))
    img = img.unsqueeze(0)
    _, pred = torch.max(model(img.to(DEVICE)), dim = 1)
    pred = pred.squeeze(0).cpu().numpy()

    return pred
    

def select_random_images(directory, num_images):
    files = os.listdir(os.path.join(directory, 'images'))
    num_images = min(num_images, len(files))
    selected_images = random.sample(files, num_images)
    image_paths = [os.path.join(directory, "images", image) for image in selected_images]
    original_mask_paths = [os.path.join(directory, "masks", image) for image in selected_images]
    #print(len(image_paths), image_paths)
    #print(len(original_mask_paths), original_mask_paths)
    return image_paths, original_mask_paths


def main():
    st.title("Segmentation Model Inference")

    # Model selection
    model_option = st.selectbox("Select Model", ["f12", "f23", "f31"]) # Add your model names here

    # Load model based on selection
    if model_option == "f12":
        model_path = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_FocalLoss_"+"f12.pth"
    elif model_option == "f12":
        model_path = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_FocalLoss_"+"f23.pth"
    else:
        model_path = "D:\\dissertation ideas\\pannuke materials\\multi-class-segmentation\\UNet_ResNet50BackEnd_ImageNetWeights_FocalLoss_"+"f31.pth"

    model = smp.Unet(
        encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,                      # model output channels (number of classes in your dataset)
        activation='softmax'
    ).to(DEVICE)
    load_checkpoint(torch.load(model_path), model)
    model.eval()
    number = int(st.number_input('How many images do you want to test on? ', format="%f"))

    test_image_paths, test_original_mask_paths = select_random_images(TEST_IMAGES_FOLDER, number)
    # Randomly select 5 images from test data
    #test_image_paths = ["image1.png", "image2.png", "image3.png", "image4.png", "image5.png"]  # Provide paths to your test images
    
    i = 0
    for image_path in test_image_paths:
        
        st.subheader(f"Image: {i+1}")

        # Perform inference
        predicted_mask = perform_inference(model, image_path)
        original_image = Image.open(image_path)
        original_mask = Image.open(test_original_mask_paths[i])
        fig, axes = plt.subplots(1, 3, figsize=(25, 5))
        # Display original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        # Display original mask
        axes[1].imshow(original_mask)
        axes[1].set_title("Original Mask")
        axes[1].axis('off')
        # Display predicted mask
        axes[2].imshow(predicted_mask) 
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        st.pyplot(fig)

        i+=1

if __name__ == "__main__":
    #select_random_images(TEST_IMAGES_FOLDER, 5)
    main()