import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing (scaling features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the initial model
model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(model, 'iris_classification_model.pkl')

# Streamlit App

st.set_page_config(layout="wide")
st.text("By Hareetima Sonkar")
st.text("Task 3")
st.title("Iris Classification")

# Sidebar
st.sidebar.header("Choose Model Parameters")

# Model selection dropdown
model_option = st.sidebar.selectbox(
    "Choose Model",
    ("Initial SVC Model", "Improved SVC Model")
)

if model_option == "Improved SVC Model":
    # Load the improved model
    model = joblib.load('iris_classification_model.pkl')
    st.sidebar.text("Using Improved SVC Model")

# Sidebar for new data input
st.sidebar.header("Input New Data")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

new_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predictions
if st.sidebar.button("Predict"):
    scaled_new_data = scaler.transform(new_data)
    prediction = model.predict(scaled_new_data)
    predicted_species = iris.target_names[prediction[0]]
    st.write(f"Predicted Species: **{predicted_species}**", unsafe_allow_html=True)

# Model Evaluation
st.header("Model Evaluation")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Test Accuracy: **{accuracy:.2f}%**", unsafe_allow_html=True)

# Model Parameters
st.header("Model Parameters")
st.write(model.get_params())
