# Telecom Customer Churn Prediction



## 1. Project Overview

Customer churn occurs when a customer stops using a company's services. Predicting churn helps telecom companies identify customers who may leave and take proactive steps to improve customer retention.

The objective of this project is to develop a machine learning model that predicts whether a telecom customer is likely to churn.

The project involves data exploration, feature engineering, model training, hyperparameter tuning, model evaluation, and inference on customer data.



## 2. About the Dataset

The project uses a telecom customer churn dataset divided into training and test files.

- **Training dataset:** `churn-bigml-80.csv`
- **Test dataset:** `churn-bigml-20.csv`

Important features include:

- **Customer information:** State and area code.
- **Service information:** International plan and voice mail plan.
- **Usage information:** Day, evening, and night call minutes and calls, along with international usage.
- **Account information:** Customer service calls and account length.
- **Target variable:** `Churn`, indicating whether a customer left the service.

###

The training dataset was used for model development and validation, while the supplied test dataset was used to evaluate the selected model.



## 3. Installation

###

### Step 1: Clone or download the project

Download the project ZIP and extract it to your computer.

### Step 2: Open the project directory

Open a terminal in the extracted `telecom_churn_project` directory.

### Step 3: Create a virtual environment
```
python -m venv .venv
```

Activate the environment on Windows:
```
.venv\Scripts\activate
```

### Step 4: Install dependencies
```css
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Add the datasets

Place the following files in the `data/` directory:
```
data/
├── churn-bigml-80.csv
└── churn-bigml-20.csv
```

### Step 6: Run the Jupyter Notebook

Open `notebooks/telecom_churn_project.ipynb` and execute the cells from top to bottom.

The notebook performs data exploration, model training, hyperparameter tuning, evaluation, and saving of the selected model artifacts.

### Step 7: Make predictions

After running the notebook and saving the model artifacts, use the reusable inference function in `src/predict.py` to predict churn for new customer records.

The function returns the estimated churn probability and the predicted churn class.

## 4. Conclusion

This project developed a machine learning workflow to identify telecom customers who may be likely to leave the service.

The key achievements were:

- Performed exploratory data analysis to understand customer usage patterns and churn distribution.
- Prepared the data and removed redundant features.
- Trained and compared Logistic Regression, Random Forest, and XGBoost models.
- Used cross-validation and hyperparameter tuning to improve model selection.
- Evaluated models using precision, recall, F1-score, confusion matrices, ROC-AUC, and PR-AUC.
- Selected a tuned XGBoost model and a probability threshold based on the trade-off between identifying churners and generating false alarms.
- Evaluated the selected model on the supplied test dataset.
- Saved the trained model and supporting artifacts for reusable inference.

On the explored test dataset, the selected model achieved approximately **94% accuracy**, **80% recall for churners**, and **80% F1-score for churners** at the selected threshold of 0.30.

These results indicate that the model can identify many customers who churn, while still making some incorrect predictions. Because the test dataset was explored during development, these scores should be considered exploratory rather than a completely independent final performance estimate.

Overall, this project demonstrates an end-to-end machine learning workflow, from raw customer data to a reusable churn prediction function, which could support customer retention strategies.
