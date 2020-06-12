# Credit_Risk_Modelling_PD_Model
The aim is to build a Logistic Regression Model to determine Credit Scores and The Probability of Default of a loan applicant.

# Dataset
The dataset is taken from [Kaggle.](https://www.kaggle.com/wendykan/lending-club-loan-data)
It contains data collected from 2260668 loan applicants. Using this dataset, a logistic regression model is built that determines the probability that a particular applicant will pay the loan in full.
The coefficients of the model are then used to build a scorecard that helps to determine the credit score of an applicant.
The dataset is too big for the repo so it is not included. Feel free to download it from the link above. 

# Initial Data Preprocessing (Initial Preprocessing.ipynb)
In this phase, a few features of the dataset are explored and cleaned (filling missing values, taking care of data types etc.).
At the end of this phase, the Dataset is split column wise into seperate .csv files in a folder called **Column** inside the current working directory.
Since, the size of the **Columns** directory is too big for this repository, it is not included.
Feel free to run all the cells of the notebook to obtain the .csv files.

# Targets and Features
- Targets (Dependent Variable): loan_status
- Features (Independent Variables):
  - Discrete Features
    - grade
    - home_ownership
    - addr_state
    - verification_status
    - purpose
    - initial_list_status
  - Continuous Features
    - term
    - emp_length
    - mths_issue_d
    - int_rate
    - funded_amnt
    - annual_inc
    - mths_earliest_cr_line
    - installment
    - delinq_2yrs
    - inq_last_6mths
    - open_acc
    - pub_rec
    - total_acc
    - acc_now_delinq
    - mths_since_last_delinq
    - mths_since_last_record
    - dti
### Pre-processing Discrete Features (PD Model Preprocessing - Discrete Variables.ipynb)
- Explore the Feature.
- Split each feature into its categories using One-Hot Encoding.
- Calculate Weight of Evidence and Infromation Value for each category.
- Group categories based on their WOE. 
- Append the grouped categories into the final training dataframe.
- If IV is less than 0.02 the entire feature is not considered in the final PD Model. Thus, note IV for each feature.

### Pre-processing Continuous Features (PD Model Preprocessing - Continuous Variables.ipynb)
- Explore the Feature.
- Do Fine Classing to convert each continuous feature into categorical feature.
- Split each feature into its fine classed categories using One-Hot Encoding.
- Calculate Weight of Evidence and Infromation Value for each category.
- Group categories based on their WOE (Coarse Classing).
- Append the grouped categories into the final training dataframe.
- If IV is less than 0.02 the entire feature is not considered in the final PD Model. Thus, note IV for each feature.

# PD Model Data Preparation (PD Model Data Preparation.ipynb)
In this step, we define the train and test set to be passed in the model.
- Features: All categorical features that have more than 0.02 IV.
- Targets: loan_status column, split into 2 categories - Default or Not_Default.

# PD Model Training (PD_Model Training - Google Colab.ipynb)
The size of the Dataset may make it computationally expensive to train a Logistic regression Model locally.
This is why, this task is performed in google colab, and the model is saved using pickle.

# PD Model Evaluation (PD Model and Performance.ipynb)
The following parameters are used to evaluate the performance of the model.
- F1 Score: 0.9326
- AUROC Curve: 0.7518
- Gini Coefficient: 0.5036
- Kolmogorov-Smirnov Coefficient: 0.3518 (p-value: 0.0)

# Building the Scorecard and Determining the Credit Scores (PD Model Scorecard.ipynb)
Using the PD Model Coefficients, whole number points for each categorical feature is calculated.
This is done because it will be easier to calculate credit score, if each category has some points associated with it.
Then for each applicant the points for their respective category is summed, and thus, credit rating can be calculated by any bank teller with a simple calculator!
The coefficients are scaled in such a way that the sum of the minimum coefficients for each feature result in 300.
Similarly the sum of the maximum coefficients for each feature result in 850. This will ensure that the credit rating of each applicant lies between 300-850.

# scorecard.csv
This file contains all the categories and their whole number points associates with them.

# pd_df.csv
This file contains all the credit ratings and Probability of Not Defaults for each loan applicant in the test set, along with the actual status of their loans (Default=0, Not Default=1)

# Dependancies
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- pickle
- scipy
