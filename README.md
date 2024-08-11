# Credit Card Fraud Detection Model

# Methodology
![Capture](https://github.com/user-attachments/assets/1d367139-8cb3-46d5-8bee-0b7e2402ca99)



# Description
**Dataset:** The dataset used for this project is loaded from a CSV file named **creditcard.csv.**

**Data Preprocessing:**

**Balancing the Dataset:** The dataset was highly imbalanced, with far fewer fraudulent transactions compared to legitimate ones.
To address this, an equal number of legitimate transactions were randomly sampled to match the number of fraudulent transactions.

**Feature Scaling:** Standardization was applied to the features to improve the performance of the Logistic Regression model.

**Model:** Logistic Regression with a maximum of 1000 iterations using the 'lbfgs' solver.

**Training Accuracy:** 94.85%

**Test Accuracy:** 93.90%

# Input / Output

**Input:** The user enters all the transaction features as comma-separated values into the input field.

**Output:** The model predicts whether the transaction is legitimate (0) or fraudulent (1).

# Screenshot of the Interface

![Blank 2 Panel Landscape Comic Strip](https://github.com/user-attachments/assets/16ef5656-9f60-431c-b58b-04d55931548e)


#  Installation
To run this project locally, follow these steps:

1.Clone the repository.

2.Ensure you have the necessary dependencies installed:
**pip install pandas scikit-learn streamlit numpy**

3.Run the Streamlit app:

**streamlit run your_script_name.py**

# Usage
Launch the Streamlit app.

Enter all the transaction features as comma-separated values in the text input field.

Click the **Submit** button to see if the transaction is legitimate or fraudulent.

# About the Data
The dataset contains the following columns:

V1, V2, ..., V28: Principal components obtained using PCA (Principal Component Analysis) from the original feature set.

**Amount:** The transaction amount.

**Class:** The label indicating if a transaction is legitimate (0) or fraudulent (1).

# Model Performance
**Training Data Accuracy:** 94.85%

**Test Data Accuracy:** 93.90%

# Acknowledgments

**Dataset Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
