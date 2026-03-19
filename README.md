# 💻 Laptop Price Prediction using Polynomial Regression

## 📌 Project Overview

This project focuses on predicting laptop prices using a machine
learning pipeline built from scratch. It involves data preprocessing,
feature engineering, normalization, and training a regression model
using gradient descent.

The dataset contains specifications such as company, RAM, CPU, GPU,
memory, and screen resolution, which are used to estimate the price of
laptops.

------------------------------------------------------------------------

## 📂 Dataset

-   Dataset: Laptop Price Dataset
-   Features include:
    -   Company
    -   TypeName
    -   Screen Resolution
    -   CPU
    -   RAM
    -   Memory
    -   GPU
    -   Operating System
    -   Weight
    -   Price (Target Variable)

------------------------------------------------------------------------

## ⚙️ Workflow

### 1. Data Preprocessing

-   Removed missing values
-   Dropped unnecessary columns
-   Handled inconsistent values ('?')
-   Converted:
    -   Ram → numerical (GB to float)
    -   Weight → float (removed kg)
    -   Inches → float

------------------------------------------------------------------------

### 2. Target-Based Label Encoding

-   Applied mean price encoding for categorical variables
-   Each category replaced with average price ranking
-   Stored encodings using pickle for reuse

------------------------------------------------------------------------

### 3. Feature Selection

-   Used correlation matrix
-   Selected features with correlation \> 0.5

Final Features: - TypeName - ScreenResolution - Cpu - Ram - Memory - Gpu

------------------------------------------------------------------------

### 4. Feature Engineering

-   Applied Polynomial Features (degree = 2)
-   Captures feature interactions

------------------------------------------------------------------------

### 5. Normalization

-   Used StandardScaler
-   Mean = 0, Standard deviation = 1

------------------------------------------------------------------------

### 6. Train / CV / Test Split

-   Train: 70%
-   Cross Validation: 20%
-   Test: 10%

------------------------------------------------------------------------

### 7. Model Training

Implemented Linear Regression using Gradient Descent

Loss Function: - Mean Squared Error (MSE)

Optimization: - Iterative parameter updates until convergence

------------------------------------------------------------------------

## 📊 Visualization

-   Histogram of prices
-   Distribution plots
-   Log transformation
-   Correlation heatmap

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Pickle

------------------------------------------------------------------------

## 🚀 How to Run

    git clone https://github.com/AnanyaChattarjee/laptop__price_prediction.git
    cd laptop__price_prediction/project_root
    ./lib_install.sh
    python project_root/training/train.py

------------------------------------------------------------------------

## 📌 Future Improvements

-   Use advanced models (Random Forest, XGBoost)
-   Hyperparameter tuning
-   Deploy using Streamlit or FastAPI

------------------------------------------------------------------------

## 👩‍💻 Author

Ananya Chatterjee\
B.Tech AI & Data Science
