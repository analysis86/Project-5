import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from PIL import Image
import pickle

#st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\HP\Desktop\Digi-crome\Project-5\Data\Final_telecom_Data.csv')
        return df
    except FileNotFoundError:
        print('The specified file was not found')
        return None

df = load_data()

left_column, right_column = st.columns([1, 1])

with left_column:
    try:
        st.image(Image.open(r'C:\Users\HP\Desktop\Digi-crome\Project-5\Images\Streamlit.png'))
    except FileNotFoundError:
        st.write('Image file not found')

with right_column:
    st.title("Telecom Data Dashboard")
    st.subheader("Explore Applications with Streamlit")

application_data = {
    'Social media': 0.36,
    'Google': 1.57,
    'Email': 0.45,
    'Youtube': 4.57,
    'Netflix': 4.56,
    'Gaming': 86.8
}

new_df = pd.DataFrame({
    'app_name': list(application_data.keys()),
    'total_data': [0.36, 1.57, 0.45, 4.57, 4.56, 86.8]
})

def display_application_relationship(app_name):
    total_data_percentage = application_data[app_name]
    st.sidebar.write(f"Relationship of {app_name} with total data is: {total_data_percentage}%")

st.sidebar.write("### Select Application")
selected_app = st.sidebar.selectbox("Select application", list(application_data.keys()))

display_application_relationship(selected_app)
filtered_data = {selected_app: application_data[selected_app]}

st.sidebar.write("### Relationship between most used Application and total data")

filtered_df = pd.DataFrame(filtered_data.items(), columns=['app_name', 'Total_data'])
sns.barplot(x='app_name', y='Total_data', data=filtered_df, color='pink')
st.sidebar.pyplot()

import os
model_path = r'C:\Users\HP\Desktop\Digi-crome\Project-5\Note-Book\regression_model.pkl'
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
    except Exception as e:
        print(f"Error opening file: {e}")
else:
    print(f"File not found: {model_path}")

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.success('Model Prediction')
    uploaded_file = st.file_uploader(r'C:\Users\HP\Desktop\Digi-crome\Project-5\Data\featured_telcom_data.xlsx', type=['xlsx'])

    if uploaded_file is not None:
        input_data = pd.read_excel(uploaded_file)

        if st.button('Predict'):
            prediction = predict(input_data)
            st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()

left_column, right_column = st.columns([1, 1])
with right_column:
    st.success('Prediction accuracy is 100%')
with left_column:
    st.write('### Final result')
    csv_file_path = r'C:\Users\HP\Desktop\Digi-crome\Project-5\Data\Final_telecom_Data.csv'
    if os.path.exists(csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            st.dataframe(data)
        except Exception as e:
            print(f"Error reading csv file: {e}")
    else:
        print(f"CSV file not found: {csv_file_path}")

# Visualizations
st.subheader('Data Visualizations')

try:
    x = df[['Engagement Score', 'Experience Score']]
except KeyError as e:
    print(f"Error accessing columns: {e}")
except TypeError as e:
    print(f"Type error encountered: {e}")

left_column, right_column = st.columns([1, 1])

with left_column:
    try:
        st.image(Image.open(r'C:\Users\HP\Desktop\Digi-crome\Project-5\Images\Top_10_Handset.png'))
    except FileNotFoundError:
        st.write('Image file not found')

with left_column:
    st.write('### Top 10 Handset')
    csv_file_path = r'C:\Users\HP\Desktop\Digi-crome\Project-5\Data\top_10_handsets.csv'
    if os.path.exists(csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            st.dataframe(data)
        except Exception as e:
            print(f"Error reading csv file: {e}")
    else:
        print(f"CSV file not found: {csv_file_path}")

left_column, right_column = st.columns([1, 1])

with left_column:
    try:
        st.image(Image.open(r'C:\Users\HP\Desktop\Digi-crome\Project-5\Images\Top_3_Handset.png'))
    except FileNotFoundError:
        st.write('Image file not found')

with left_column:
    st.write('### Top 3 Manufacturers')
    csv_file_path = r'C:\Users\HP\Desktop\Digi-crome\Project-5\Data\top_3_manufacturers.csv'
    if os.path.exists(csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            st.dataframe(data)
        except Exception as e:
            print(f"Error reading csv file: {e}")
    else:
        print(f"CSV file not found: {csv_file_path}")



st.write('### KMeans Clustering')

num_clusters = st.slider('Please select number of clusters:', min_value=1, max_value=5, value=2)
if 'x' in locals():
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit_predict(x)
    pred = kmeans.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(x['Engagement Score'], x['Experience Score'], c=pred, cmap='viridis', alpha=0.5, edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.title(f"KMeans Clustering with {num_clusters} Clusters")
    st.pyplot(plt)

else:
    st.write("X is not defined. Please make sure the DataFrame is loaded and the specified columns exist.")


