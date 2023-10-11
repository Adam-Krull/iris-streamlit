import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def viz_results(data, labels):
    
    for combo in itertools.combinations(data.columns, 2):
        
        name = combo[0].replace('_', ' ').capitalize()
        nametoo = combo[1].replace('_', ' ').capitalize()
        
        fig, ax = plt.subplots()
        ax.scatter(data[combo[0]], data[combo[1]], c=labels)
        ax.set_title(f'Combination of {name} and {nametoo}')
        ax.set_xlabel(f'{name}')
        ax.set_ylabel(f'{nametoo}')
        st.pyplot(fig)
        
        
def scale_data(data):
    
    mms = MinMaxScaler()
    
    return mms.fit_transform(data)


def fit_kmeans(data, k, seed=42):
    
    model = KMeans(n_clusters=k, random_state=seed)
    
    data = scale_data(data)
    
    model.fit(data)
    
    return model.labels_


def form_unpacker(response):
    
    cols_to_keep = []
    
    for i, item in enumerate(response):
        
        if item:
            
            cols_to_keep.append(i)
            
        else:
            
            continue
            
    return cols_to_keep        


def main():
    
    st.title("One dataset to rule them all")
    
    st.subheader("Clustering the Iris dataset with KMeans")
    
    st.divider()
    
    df = sns.load_dataset('iris')
    
    x_df = df.drop(columns=['species'])
    
    st.header("Acquire and prepare")
    
    st.markdown("The data was acquired using the seaborn library. The target variable, species, was dropped from the dataset. Below, you can see the first few observations.")
    
    st.dataframe(x_df.head())
    
    st.divider()
    
    st.header("Modeling")
    
    st.markdown("The data is scaled using the MinMaxScaler before clustering. The MinMaxScaler scales all values to be from 0-1, while preserving the shape of the original data. It's important to scale data before using the KMeans clustering algorithm, which uses distance to assign data points to clusters. The KMeans algorithm will define however many clusters you specify.")
    
    with st.form(key='feature_selection'):
        
        st.write("Which features would you like to include in your clusters?")
        
        form_items = [st.checkbox(col) for col in x_df.columns]
        
        st.divider()
        
        st.write("How many clusters would you like to create?")
        
        num_clusters = st.text_input("Enter a value for k:")
        
        submit = st.form_submit_button()
        
    cols = form_unpacker(form_items)
    
    filtered_df = df.iloc[:, cols]
    
    try:
        
        num_clusters = int(num_clusters)
        
    except:
        
        pass
        
    if submit:    
        
        labels = fit_kmeans(filtered_df, num_clusters)    
        
        viz_results(filtered_df, labels)
        
    
if __name__ == '__main__':
    
    main()