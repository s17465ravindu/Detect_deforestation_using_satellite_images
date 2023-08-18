import streamlit as st
import pandas as pd
import numpy as np
#import rasterio 
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
#from sklearn.metrics import precision_score, recall_score 
#import pickle
#import matplotlib.pyplot as plt
#import os
#from PIL import Image

 #load the model
'''@st.cache(persist=True)
def prediction(mean_ndvi_value):
    file_name = 'deforestaion.pickle'
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([[mean_ndvi_value]])
    return pred_value

#Calculate mean advi
def calculate_mean_ndvi(tiff_file):
     with rasterio.open(tiff_file) as src:

        ndvi_data = src.read(1)

        deforested_mask = ndvi_data < 0.33
        non_deforested_mask = ~deforested_mask

        masked_ndvi = np.ma.masked_array(ndvi_data, mask=non_deforested_mask)

        return masked_ndvi.mean()
     
#Show masekd image
def display_deforested_area(tiff_file):
    with rasterio.open(tiff_file) as src:
        ndvi_data = src.read(1)

        deforested_mask = ndvi_data < 0.33

        masked_deforested = np.ma.masked_array(ndvi_data, mask= deforested_mask)
        cmap = plt.cm.RdYlGn 
        cmap.set_bad(color='brown')
        plt.figure(figsize=(7, 7))
        plt.imshow(masked_deforested,cmap=cmap)

def visualize_tiff_images(upload):
    with rasterio.open(upload) as src:
        #tif_data = src.read(1)
        tif_data = Image.open(upload)
        st.image(tif_data)'''

def main():
    st.title("Deforestation Detection - WebApp")
    st.markdown("Deforestation entails the extensive removal of trees, primarily due to human activities like logging and agriculture, this cause ecological imbalance and climate repercussions. Leverage this advanced model to swiftly determine the existing condition of a specified area – whether it has undergone deforestation or remains unaffected")

    st.sidebar.title("Upload ⬆️ ")
    st.sidebar.markdown("Upload NDVI satellite image of the area that you want to classify")
    st.sidebar.markdown("(Only TIF format images are accepted)")
    

    #my_upload = st.sidebar.file_uploader("Upload an image", type=["tif"])
   ''' visualize_tiff_images(my_upload)

    #if my_upload is not None:
        #visualize_tiff_images(my_upload)
    #else:
       # st.write("come to this 2")'''





if __name__ == '__main__':
    main()
