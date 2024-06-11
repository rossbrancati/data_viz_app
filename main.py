# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import pdist, cdist




# Set the page configuration
st.set_page_config(page_title='Data Visualization for Cluster Analysis',
                   layout='wide',
                   page_icon='📈')

# Title
st.title('Data Visualization for Cluster Analysis')
st.header('Dashboard for exploring data for cluster analysis')
st.header('Part 1: Data Exploration')
st.subheader('1) Select a .csv file')


# Assigning working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Assign folder path
# folder_path = f"{working_dir}/data"

# List files in data folder
# files_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Instead of using a dropdown, have the user upload their csv file
selected_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Only run next section if file is selected
if selected_file is not None:

    # Read file as pandas df
    df_og = pd.read_csv(selected_file)

    # Subheader - select which group you want to analyze (Healthy, Symptomatic, or All)
    st.subheader('2) Select groups to analyze')

    # Select the column with the groups
    group_id_col = st.selectbox('Select the column with group identifiers, then select which groups you would like to explore', options=df_og.columns.tolist(), index=0)

    # Selecting groups of interest for analysis
    selected_groups = []
    unique_groups_all = df_og[group_id_col].unique()
    for group in unique_groups_all:
        if st.checkbox(group):
            selected_groups.append(group)

    # Assign dataframe based on groups
    df = df_og[df_og[group_id_col].isin(selected_groups)]

    # Subheader
    st.subheader("3) Select variables of interest")

    # Creates 2 columns, one for displaying the data preview and another for displaying the selected columns
    col1, col2, col3 = st.columns(3)

    # Create a list of the columns of the dataframe
    columns = df.columns.tolist()

    # Left side histogram (var1)
    with col1:
        var1 = st.selectbox('Select first variable', options=columns + ['None'], index=0)

    # Middle histogram (var2)
    with col2:
        var2 = st.selectbox('Select second variable', options=columns + ['None'], index=0)

    # Right side histogram (var3)
    with col3:
        var3 = st.selectbox('Select third variable', options=columns + ['None'], index=0)


    # subheader
    st.subheader('4) Visualize distributions and scatterplots')

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))

    # Plot first variable if selected
    if var1 != 'None':
        sns.histplot(df[var1], ax=axes[0], color='dodgerblue')
        # axes[0].set_title(f'Distribution of {var1}')
        axes[0].set_title(var1)
    else:
        axes[0].set_visible(False)

    # Plot second variable if selected
    if var2 != 'None':
        sns.histplot(df[var2], ax=axes[1], color='firebrick')
        # axes[1].set_title(f'Distribution of {var2}')
        axes[1].set_title(var2)
    else:
        axes[1].set_visible(False)

    # Plot third variable if selected
    if var3 != 'None':
        sns.histplot(df[var3], ax=axes[2], color='forestgreen')
        #axes[2].set_title(f'Distribution of {var3}')
        axes[2].set_title(var3)
    else:
        axes[2].set_visible(False)

    # Display the plots
    plt.tight_layout()
    st.pyplot(fig)

    # Scatter plots
    fig_2, axes_2 = plt.subplots(1, 2, figsize=(8, 4))

    # Plot first variable if selected
    if var1 != 'None' and var2 != 'None':
        sns.scatterplot(x=df[var2], y=df[var1], ax=axes_2[0], hue=df[group_id_col])
        # axes_2[0].set_title(f'Scatterplot of {var1} and {var2}')
        axes_2[0].set(xlabel=f'{var2}', ylabel=f'{var1}')
        axes_2[0].legend([],[], frameon=False)
    else:
        axes_2[0].set_visible(False)

    if var2 != 'None' and var3 != 'None':
        sns.scatterplot(x=df[var2], y=df[var3], ax=axes_2[1], hue=df[group_id_col])
        # axes_2[1].set_title(f'Scatterplot of {var3} and {var2}')
        axes_2[1].set(xlabel=f'{var2}', ylabel=f'{var3}')
        axes_2[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)


    else:
        axes_2[1].set_visible(False)

    #Display the scatterplots
    plt.tight_layout()
    st.pyplot(fig_2)

    # Create 3D scatter plot if all three variables are selected
    if var1 != 'None' and var2 != 'None' and var3 != 'None':
        fig_3d = px.scatter_3d(df, x=var1, y=var2, z=var3,
                               color=df[group_id_col],
                               size_max=18,
                               title=f'3D Scatter Plot of {var1}, {var2}, and {var3}')
        fig_3d.update_layout(width=1000, height=600)
        st.plotly_chart(fig_3d)

    # Part 2: Clustering models
    st.header('Part 2: Clustering Models')
    st.subheader('1) Select features for clustering')

    # Selecting variables of interest for clustering
    selected_columns = []
    for column in df.columns:
        if st.checkbox(column):
            selected_columns.append(column)


    # Subheader
    st.subheader('2) Select clustering model and # of clusters')

    if selected_columns:
        # Select clustering model
        model_choice = st.selectbox('Model options:', options=['K-Means', 'DBSCAN', 'None'], index=0)

        if model_choice == 'K-Means':
            num_clusters = st.number_input('Number of clusters: ', min_value=1, max_value=10, value=3)

        elif model_choice == 'DBSCAN':
            st.markdown('*Epsilon: max distance between two samples or one to be considered in the neighborhood of the other')
            st.markdown('*Min. Samples: minimum number of samples in a neighborhood for a datapoint to be considered as a core point')
            epsilon = st.number_input('Epsilon: ', min_value=0.01, max_value=5.00, value=1.0)
            min_no_samples = st.number_input('Min. Samples: ', min_value=5, max_value=1000, value=10)

        # Select the data for clustering
        data_for_clustering = df[selected_columns]

        # Assign a new dataframe for visualizing clusters
        data_for_clustering_viz = df[selected_columns]

        # Preprocess the data: scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)


        # Define functions used to calculate the Dunn Index
        def calculate_intra_cluster_distances(data, labels):
            unique_clusters = np.unique(labels)
            intra_cluster_distances = []
            for cluster in unique_clusters:
                cluster_points = data[labels == cluster]
                if len(cluster_points) > 1:
                    intra_cluster_distances.append(np.max(pdist(cluster_points)))
                else:
                    intra_cluster_distances.append(0)
            return intra_cluster_distances

        def calculate_inter_cluster_distances(data, labels):
            unique_clusters = np.unique(labels)
            inter_cluster_distances = []
            for i in range(len(unique_clusters)):
                for j in range(i + 1, len(unique_clusters)):
                    cluster_i_points = data[labels == unique_clusters[i]]
                    cluster_j_points = data[labels == unique_clusters[j]]
                    inter_cluster_distances.append(np.min(cdist(cluster_i_points, cluster_j_points)))
            return inter_cluster_distances

        def dunn_index(data, labels):
            intra_cluster_distances = calculate_intra_cluster_distances(data, labels)
            inter_cluster_distances = calculate_inter_cluster_distances(data, labels)
            max_intra_cluster_distance = np.max(intra_cluster_distances)
            min_inter_cluster_distance = np.min(inter_cluster_distances)
            dunn_index_value = min_inter_cluster_distance / max_intra_cluster_distance
            return dunn_index_value



        # Run K-means clustering
        if model_choice == 'K-Means':
            st.subheader('Running K-means clustering...')

            # Run K-means
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(scaled_data)

            # Get the cluster labels
            cluster_labels = kmeans.labels_

            # Add the cluster labels to the original DataFrame
            data_for_clustering_viz['Cluster'] = cluster_labels

            # Calculate inertia
            inertia = kmeans.inertia_

            # Calculate silouhette score
            silh_score = silhouette_score(scaled_data, cluster_labels)

            # Calculate Dunn Index
            dunn_index_value = dunn_index(scaled_data, cluster_labels)

        elif model_choice == 'DBSCAN':
            st.subheader('Running DBSCAN clustering...')

            # Run DBSCAN
            dbscan = DBSCAN(eps=epsilon, min_samples=min_no_samples)

            # Fit DBSCAN
            dbscan.fit(data_for_clustering)

            # Get labels
            cluster_labels = dbscan.labels_

            # Add the cluster labels to the original DataFrame
            data_for_clustering_viz['Cluster'] = cluster_labels

            # Calculate silouhette score
            silh_score = silhouette_score(data_for_clustering, cluster_labels)

            # Calculate Dunn Index
            dunn_index_value = dunn_index(data_for_clustering, cluster_labels)


        st.subheader('3) Analyze goodness of fit')
        if model_choice == 'K-Means':
            st.markdown('*Inertia (K-Means only) measures how compact the clusters are (lower = better)')
            st.markdown('*Silhouette score measures how well clusters are separated on a scale of -1 to +1 (higher = better)')
            st.markdown('*Dunn index measures how defined and separated clusters are (higher = better)')
            # Create a dictionary with goodness of fit values, convert to df, display
            good_of_fit_dict = {'Inertia': [inertia], 'Silhouette Score': [silh_score], 'Dunn Index': [dunn_index_value]}
            good_of_fit_df = pd.DataFrame(good_of_fit_dict)
            st.dataframe(good_of_fit_df)

        elif model_choice == 'DBSCAN':
            st.markdown('*Silhouette score measures how well clusters are separated on a scale of -1 to +1 (higher = better)')
            st.markdown('*Dunn index measures how defined and separated clusters are (higher = better)')
            # Create a dictionary with goodness of fit values, convert to df, display
            good_of_fit_dict = {'Silhouette Score': [silh_score], 'Dunn Index': [dunn_index_value]}
            good_of_fit_df = pd.DataFrame(good_of_fit_dict)
            st.dataframe(good_of_fit_df)


        # Select variables for visualization
        st.subheader('4) Select variables for visualizing clustering')

        col4, col5, col6 = st.columns(3)

        with col4:
            var1_clust = st.selectbox('Select first variable', options=selected_columns + ['None'], index=0)
        with col5:
            var2_clust = st.selectbox('Select second variable', options=selected_columns + ['None'], index=0)
        with col6:
            var3_clust = st.selectbox('Select third variable', options=selected_columns + ['None'], index=0)

        # Create 3D scatter plot if all three variables are selected
        if var1_clust != 'None' and var2_clust != 'None' and var3_clust != 'None':
            fig_3d = px.scatter_3d(data_for_clustering_viz, x=var1_clust, y=var2_clust, z=var3_clust,
                                   color=data_for_clustering_viz['Cluster'],
                                   size_max=18,
                                   title=f'3D Scatter Plot of {model_choice} cluster labels')
            fig_3d.update_layout(width=1000, height=600)
            st.plotly_chart(fig_3d)




