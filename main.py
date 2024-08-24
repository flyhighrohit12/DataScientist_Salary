import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Set page config
st.set_page_config(page_title="Salary Predictor", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
file = st.sidebar.file_uploader("Upload your dataset", type=['csv'])

# Data loading function
def load_data():
    if file:
        df = pd.read_csv(file)
    else:
        # Ensure you have a default dataset or handle this case appropriately
        df = pd.read_csv("ds_salaries.csv")  
    return df

df = load_data()

tabs = ["How to Use", "Explore Data", "Train & Evaluate", "Predict"]
tab_icons = ["📖", "🔍", "🏋️‍♂️", "🎯"]
selected_tab = st.sidebar.radio("Select a tab", tabs, format_func=lambda x: f"{tab_icons[tabs.index(x)]} {x}")

# How to Use tab
if selected_tab == "How to Use":
    st.title("How to Use the Salary Predictor")
    st.markdown("""
        Welcome to the Salary Predictor app! Here's how to use it:
        
        **1. Explore Data**
        - View sample data and dataset information.
        - Explore visualizations to understand data patterns.
        
        **2. Train & Evaluate**
        - Select a regression algorithm.
        - Choose training data percentage and hyperparameters.
        - Train the model and view performance metrics.
        - Compare different models.
        
        **3. Predict**
        - Enter feature values for a new data point.
        - Select a trained model to make predictions.
        - View the predicted salary.
    """)

# Explore Data tab
elif selected_tab == "Explore Data":
    st.title("Explore Data")
    
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.subheader("Dataset Description")
    st.write(df.describe())
    
    st.subheader("Visualizations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['salary_in_usd'], kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("Salary by Experience Level")
        fig, ax = plt.subplots()
        sns.boxplot(x='experience_level', y='salary_in_usd', data=df, ax=ax)
        st.pyplot(fig)
    
    with col3:
        st.write("Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Train & Evaluate tab
elif selected_tab == "Train & Evaluate":
    st.title("Train & Evaluate Models")
    
    # Prepare data
    le = LabelEncoder()
    df['encoded_job_title'] = le.fit_transform(df['job_title'])
    df['encoded_experience_level'] = le.fit_transform(df['experience_level'])
    df['encoded_employment_type'] = le.fit_transform(df['employment_type'])
    df['encoded_company_size'] = le.fit_transform(df['company_size'])
    
    X = df[['encoded_job_title', 'encoded_experience_level', 'encoded_employment_type', 'encoded_company_size']]
    y = df['salary_in_usd']
    
    # Model selection
    model_name = st.selectbox("Select Regression Algorithm", ["Linear Regression", "Decision Tree", "Random Forest"])
    
    # Training percentage
    train_percent = st.slider("Select percentage of data for training", 50, 90, 80)
    
    # Hyperparameters
    if model_name == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 20, 5)
    
    # Train model
    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_percent/100, random_state=42)
        
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Save model
        joblib.dump(model, f"{model_name.replace(' ', '')}.pkl")
        
        # Display metrics
        st.subheader("Model Performance")
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R-squared", f"{r2:.3f}")
        col2.metric("MSE", f"{mse:.3f}")
        col3.metric("MAE", f"{mae:.3f}")
        col4.metric("RMSE", f"{rmse:.3f}")
        
        # Actual vs Predicted plot
        st.subheader("Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
        
        # Feature importance
        #if model_name != "Linear Regression":
        if model_name == "Linear Regression":
            importances = model.coef_
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        else:
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        st.pyplot(fig)

# Predict tab
# Predict tab
else:
    st.title("Predict Salary")
    
    # Load trained models
    models = {}
    for model_name in ["LinearRegression", "DecisionTree", "RandomForest"]:
        if os.path.exists(f"{model_name}.pkl"):
            models[model_name] = joblib.load(f"{model_name}.pkl")
    
    if not models:
        st.warning("No trained models found. Please train a model first.")
    else:
        # Feature input
        job_title = st.selectbox("Job Title", df['job_title'].unique())
        experience_level = st.selectbox("Experience Level", df['experience_level'].unique())
        employment_type = st.selectbox("Employment Type", df['employment_type'].unique())
        company_size = st.selectbox("Company Size", df['company_size'].unique())
        
        # Encode inputs
        le_job = LabelEncoder().fit(df['job_title'])
        le_exp = LabelEncoder().fit(df['experience_level'])
        le_emp = LabelEncoder().fit(df['employment_type'])
        le_size = LabelEncoder().fit(df['company_size'])
        
        encoded_job_title = le_job.transform([job_title])[0]
        encoded_experience_level = le_exp.transform([experience_level])[0]
        encoded_employment_type = le_emp.transform([employment_type])[0]
        encoded_company_size = le_size.transform([company_size])[0]
        
        # Model selection
        selected_model = st.selectbox("Select Model for Prediction", list(models.keys()))
        
        if st.button("Predict Salary"):
            input_data = np.array([[encoded_job_title, encoded_experience_level, encoded_employment_type, encoded_company_size]])
            prediction = models[selected_model].predict(input_data)[0]
            
            st.success(f"Predicted Salary: ${prediction:,.2f}")

st.sidebar.info("Created by Rohit Sharma")
