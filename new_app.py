## importing libraries

import streamlit as st
import requests
from io import BytesIO
import numpy as np
import os
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

## setting device for running of the model using device fruntion from torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  ###uncomment when cpu only maching
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## pre-trained weights to device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

## converting images from rgb format to L_ab format
def generate_l_ab(images): 
    lab = rgb2lab(images.permute(0, 2, 3, 1).cpu().numpy())
    X = lab[:,:,:,0]
    X = X.reshape(X.shape+(1,))
    Y = lab[:,:,:,1:] / 128
    return to_device(torch.tensor(X, dtype = torch.float).permute(0, 3, 1, 2), device),to_device(torch.tensor(Y, dtype = torch.float).permute(0, 3, 1, 2), device)

## BaseModel class declared in order to satisfy dependencies during Encoder_Decoder()
class BaseModel(nn.Module):
    def training_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return loss

    def validation_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return {'val_loss' : loss.item()}

    def validation_end_epoch(self, outputs):
        epoch_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        return {'epoch_loss' : epoch_loss}

## using the formula to calculate padding
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

## the auto-encoder model to run when provided with the weights
class Encoder_Decoder(BaseModel):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size = 3, padding = get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size = 3, padding = get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size = 3, padding = get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(512, 512, kernel_size = 3, padding = get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size = 3, padding = get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        
            nn.Conv2d(256, 128, kernel_size = 3, padding = get_padding(3)),
            nn.Upsample(size = (64,64)),
            nn.Conv2d(128, 64, kernel_size = 3, padding = get_padding(3)),
            nn.Upsample(size = (128,128)),
            nn.Conv2d(64, 32, kernel_size = 3, padding = get_padding(3)),
            nn.Conv2d(32, 16, kernel_size = 3, padding = get_padding(3)),
            nn.Conv2d(16, 2, kernel_size = 3, padding = get_padding(3)),
            nn.Tanh(),
            nn.Upsample(size = (256,256))
    )

    def forward(self, images):
        return self.network(images)

## initialising the model variable  
model = Encoder_Decoder()
to_device(model, device)

## function to load a model using pre-trained weights
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')  ##uncomment for cpu-only machines
    # checkpoint = torch.load(filepath)                    ##uncomment for cuda-available machines
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.eval()
    return model

## initialsing model_xxxx variables with pre-trained weights
# model_gen = load_checkpoint('genralised_test.pth')  ##unavailable model due to lack of resources
model_fruit = load_checkpoint('fruits_test.pth')
model_animal = load_checkpoint('animals_test.pth')
model_people = load_checkpoint('faces_test.pth')
model_land = load_checkpoint('landscapes_test.pth')

model = model_fruit # model = model_gen
to_device(model, device)

## function to convert image from L_ab format to rgb format for front-end presentation
def to_rgb(grayscale_input, ab_output, save_path=None, save_name=None):
    color_image = torch.cat((grayscale_input, ab_output), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1]
    color_image[:, :, 1:3] = (color_image[:, :, 1:3]) * 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    return color_image

## executing the auto-encoder in this function in order to provide predicted colorized version of the grayscale image
def prediction(img, m):
    if(m == "l"):
        model = model_land
    elif(m == "f"):
        model = model_fruit
    elif(m == "a"):
        model = model_animal
    elif(m == "p"):
        model = model_people
    # else:
    #     model = model_gen
    to_device(model, device)
    a = rgb2lab(img.permute(1, 2, 0))
    a = torch.tensor(a[:,:,0]).type(torch.FloatTensor)
    a = a.unsqueeze(0)
    a = a.unsqueeze(0)
    xb = to_device(a, device)
    ab_img = model(xb)
    xb = xb.squeeze(0)
    ab_img = ab_img.squeeze(0)
    return to_rgb(xb.detach().cpu(), ab_img.detach().cpu())

##################################################################################################################################

## utility functions to image to feasible formats for the prediction function to access
def transform_image(image):  # convert all images into a similar size
    test_transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    return test_transforms(image)

def convert_to_tensor(image):
    tensor = transform_image(image)
    if(tensor.shape[0] == 1):
        tensor = tensor.repeat(3, 1, 1)
    return tensor

###################################################################################################################################

## initialising the streamlit page
st.set_page_config(

    page_title="Image Colorizer",
    page_icon="photo-gallery.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Image Colorizer')

## attaching the css style sheet
with open('style.css') as f :
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

## sidebar declaration, model will be selected here
chosenModel=st.sidebar.selectbox("Select a model",("Landscape","Fruits","Animals","Faces"))

## uploading of file via uploader or url
uploaded_file = st.file_uploader("Choose a file")
image_url = st.text_input("URL : ")

if uploaded_file is None and image_url:
    try:
        response = requests.get(image_url)
        uploaded_file = BytesIO(response.content)
        print("Image successfully fetched")
    except:
        st.write("Please enter a valid URL")

if uploaded_file is None:
    st.write('Please upload an image or provide a url')

## declaration of button that in turn will call the prediction function
btn = st.button("COLORIZE")

## with updated uploaded_file, on click of button execute prediction
if(btn and uploaded_file):
    if chosenModel=='Landscape':
        m='l'
    elif chosenModel=='Fruits':
        m='f'
    elif chosenModel=='Animals':
        m='a'
    elif chosenModel=='Faces':
        m='p'
    else:
        m='x'
    img = Image.open(uploaded_file)
    width, height = img.size
    tensor = transform_image(img)
    output = prediction(tensor, m)
    result = Image.fromarray((output*255).astype(np.uint8))
    # st.image(result, caption = "colorised")
    new_result = result.resize((width, height))

    col1, col2 = st.columns(2)
    col1.header("B/W")
    col1.image(img, width=300)
    col2.header("Colorized")
    col2.image(new_result, width=300)

else:
    st.write("Upload an image")
