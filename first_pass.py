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
        
        fig, ax = plt.subplots()
        ax.scatter(data[combo[0]], data[combo[1]], c=labels)
        ax.set_title(f'Combination of {combo[0]} and {combo[1]}.')
        ax.set_xlabel(f'{combo[0]}')
        ax.set_ylabel(f'{combo[1]}')
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
    
    df = sns.load_dataset('iris')
    
    x_df = df.drop(columns=['species'])
    
    st.dataframe(x_df.head())
    
    with st.form(key='feature_selection'):
        
        st.write("Which features would you like to include in your clusters?")
        
        form_items = [st.checkbox(col) for col in x_df.columns]
        
        submit = st.form_submit_button()
        
    cols = form_unpacker(form_items)
    
    filtered_df = df.iloc[:, cols]
    
    with st.form(key='num_clusters'):
        
        st.write("How many clusters would you like to create?")
        
        num_clusters = st.text_input("Enter a value for k:")
        
        submit_two = st.form_submit_button()
        
    try:
        
        num_clusters = int(num_clusters)
        
    except:
        
        st.write("Please enter a valid integer for the number of clusters.")
        
    if submit_two:    
        
        st.write("We have entered the second stage of the application.")
        
        labels = fit_kmeans(filtered_df, num_clusters)    
        
        viz_results(filtered_df, labels)
        
    
if __name__ == '__main__':
    
    main()