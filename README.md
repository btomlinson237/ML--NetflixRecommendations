# Netflix Recommendation System

## Overview
This project builds a machine learning-based recommendation system for Netflix movies. It explores clustering techniques and similarity measures to recommend movies based on textual and numerical features.
Note: code was originally processed in Google Colab, cells may require adjustment for correct execution on local systems.

## Features
- **Data Preprocessing**: Cleans and processes movie metadata.
- **Text Analysis**: Tokenization, stopword removal, and TF-IDF vectorization.
- **Clustering Models**: Uses K-Means and Agglomerative Clustering.
- **Similarity Metrics**: Computes cosine similarity between movie features.
- **Visualization**: PCA, Silhouette Score, Dendrograms for clustering analysis.

## Technologies Used
- **Python**
- **Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `statsmodels`, `yellowbrick`

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Load the dataset (place `Netflix.csv` in the working directory).
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook Netflix_ML_Workshop.ipynb
   ```

## Usage
- Run the notebook to preprocess the dataset and build the clustering models.
- Analyze recommendations based on cosine similarity scores.
- Visualize cluster groupings and assess model performance.

