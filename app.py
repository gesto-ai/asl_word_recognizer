import streamlit as st
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys

sys.path.append("../")

from asl_sign_spelling_classification.train_v0 import * 

### weight location
weight_loc = "<>"



st.header("Welcome to Gesto AI")
st.write("Choose any image and get predicted value")

uploaded_file = st.file_uploader("Choose an image...")



def load_model(device = 0):
    model = PL_resnet50().eval().cuda(device=device)
    pretrained_model = PL_resnet50.load_from_checkpoint(checkpoint_path =weight_loc).eval().cuda(device=0)
    return pretrained_model

def predict_on_image(img, device):
    img = data_transform(img)
    img = img.float().unsqueeze(0).cuda(device)
    y_hat = pretrained_model(img).data.cpu()
    y_hat = torch.argmax(y_hat,dim=1)
    return y_hat.cpu().detach().numpy().tolist()


if uploaded_file is not None:
    pretrained_model = load_model(0)
    image = Image.open(uploaded_file)	
    st.image(image, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    res = predict_on_image(uploaded_file)
    st.write(res)
    # im = imgGen2(uploaded_file)	

    
