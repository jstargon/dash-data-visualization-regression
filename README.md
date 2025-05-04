# https://dash-data-visualization-regression.onrender.com

# 🧮 Interactive Regression Dashboard  
**A Web App for Exploratory Data Analysis, Model Training, and Prediction**

## 📌 Overview
This project is an **interactive web dashboard built with Dash** that allows users to:  
- 📂 **Upload any CSV dataset**  
- 🎯 **Select a target variable for regression modeling**  
- 📊 **Generate bar charts for categorical features and correlation matrices**  
- 🧪 **Train a regression model using any combination of features**  
- 🤖 **Predict the target variable using the trained model**  

The goal is to create a **streamlined tool for exploratory data analysis and regression modeling—no coding required by the end user.**

---

## 🏗️ Technologies Used
- 🐍 **Python**
- 📊 **Dash**
- 🔬 **scikit-learn**
- ⚡ **XGBoost**
- 🐼 **pandas**
- 💻 **HTML/CSS**

---

## 📂 File Structure
This repository contains:

### 🔹 **Main Application**
- 📄 [`app.py`](./app.py) → Main Dash application script

### 🔹 **Supporting Files**
📂 **assets/**  
- 📄 `styles.css` → Custom CSS styling for the dashboard

---

## 🔍 Key Features
- **Dataset Upload** → Accepts CSV files for dynamic data input
- **Target Selection** → Dropdown menu to choose regression target
- **EDA Visualizations** → Automatically generates:
  - Bar plots for categorical variables
  - Correlation strength of numerical variables with the target variable
- **Custom Model Training** → User can select which features to include in model
- **Prediction Interface** → Input new values and get predictions from the trained model

---

## ⚡ Skills Demonstrated
✅ **Dash App Development**  
✅ **Interactive Data Visualization**  
✅ **Regression Model Implementation (scikit-learn, XGBoost)**  
✅ **Feature Selection Integration**  
✅ **Front-End Styling with CSS in Dash**  

---

## 🔄 Future Improvements & Lessons Learned
This project served as a **foundational step in deploying machine learning apps on the web.** Possible improvements for future iterations:  
- 📌 Adding a segement for displaying variable statistics 
- 📌 Enhance error handling for invalid data uploads  
- 📌 Include more visualization options (e.g., boxplots, histograms)
- 📌 Adding more models that the user can train with parameter customization
- 📌 Allow exporting trained models  

Building this app really deepened my understanding of how to integrate machine learning pipelines into an interactive dashboard. I learned a lot—from getting more comfortable with **Git**, to exploring how the **front-end and back-end work together for a full-stack experience**, and figuring out how to **deploy a web app**.

I also gained insight into **different regression models, their theories, use cases, and advantages,** along with techniques for **hyperparameter optimization, feature tuning, and improving model performance**. On top of that, I learned more about **data preprocessing,** further building my knowledge of the data analysis process as a whole.

---

## 🎯 Acknowledgments

This project was originally developed as part of a team project with my friends [Srinesh Selvraj](https://github.com/srineshselvaraj), and [Parth Kabaria](https://github.com/parkab) at NJIT. 

I’d like to thank them for their contributions to the initial development of this project especially for their work on the front-end and intial deployment. The initial deployment can be found [here](https://milestone4group9.onrender.com/)

Since then, I have independently made additional modifications and improvements to this repository inspired by my interest in **making machine learning tools more accessible and easy to use,** and my goal of creating **hands-on, adaptive, insightful, interactive data science applications.** Some noteable modifications and improvements include the following:
- 📌 Adding more comments to explain code functionality
- 📌 Formatting, refactoring, and organizing code for greater readibility
- 📌 Creating Model Selection
- 📌 Implementing classification models and incorporating more model evaluation components
- 📌 Fixing errors and making the application more fault-tolerant for untidy datasets
- 📌 Displaying optimal hyperparameters for grid-search
- 📌 Displaying features and their respective coefficients (and the intercept) for linear regression
- 📌 Refining the user interface to improve aesthetics and appeal

