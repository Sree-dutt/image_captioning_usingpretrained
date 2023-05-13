#importing the necessary libraries
import numpy as np
import torch
import transformers
import streamlit as st
from PIL import Image
from transformers import BlipProcessor,BlipForConditionalGeneration
##NOW WE NEED TO LOAD THE PRE TRAINED BLIP MODEL
image_processor=BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model=BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
st.title("Image Caption Generator") ##creating the title in the streamlit website
image_upload=st.file_uploader('Please upload the image') ##creating an option to uplaod the image
if image_upload is not None:
    image=Image.open(image_upload) ##Opening the uploaded image
    st.image(image)
    model_input=image_processor(image,return_tensors="pt") ##taking the image and doing feature extraction
    model_output=model.generate(**model_input) ##taking the extracted features and generating a caption
    st.write('The caption is : ',image_processor.decode(model_output[0],skip_special_tokens=True))