import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Hierarchical Clustering App 🗂️")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select columns for clustering
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_cols = st.multiselect("Select columns for clustering", numeric_cols, default=numeric_cols)

    if selected_cols:
        X = df[selected_cols]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        df['Cluster'] = clustering.fit_predict(X_scaled)

        st.write("Clustered Data:", df)

        # Plot clusters (pairplot)
        st.write("Cluster Plot:")
        sns.pairplot(df[selected_cols + ['Cluster']], hue='Cluster', palette='Set2')
        plt.tight_layout()
        st.pyplot(plt)
