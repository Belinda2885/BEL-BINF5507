# import all necessary libraries here

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
#from sklearn.impute import SimpleImputer




# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    data = data.copy()  # avoid modifying original

    if strategy in ['mean', 'median']:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                if strategy == 'mean':
                    data[col].fillna(data[col].mean(), inplace=True)
                elif strategy == 'median':
                    data[col].fillna(data[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in data.columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        raise ValueError("Unsupported strategy: choose 'mean', 'median', or 'mode'.")

    return data


    


# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    data_no_duplicates = data.drop_duplicates()
    return data_no_duplicates
# Note: The function remove_duplicates is not used in the simple_model function, but it can be used separately if needed.
# Uncomment the line below if you want to use the remove_duplicates function in the simple_model function
    remove_duplicates(data)  # Uncomment this line to use the function
pass

# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be either 'minmax' or 'standard'.")

    # Select only numerical columns for normalization
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data
pass

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation above the threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Drop redundant features
    data_reduced = data.drop(columns=to_drop)
    
    return data_reduced
pass

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """
    # Step 1: Remove columns with missing data
    input_data = input_data.dropna(axis=1, how='any')

    # Step 2: Split the data into features and target
    X = input_data.drop(columns=['target'])
    y = input_data['target']

    # Step 3: Encode categorical features using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Step 4: Split the data into training and testing sets
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Step 5: Scale the features using StandardScaler
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Step 6: Instantiate and fit a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Step 7: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model using accuracy score and classification report
    accuracy = accuracy_score(y_test, y_pred)
    
    if print_report:
        print(classification_report(y_test, y_pred))

    # Print the accuracy score
    print(f"Accuracy: {accuracy:.2f}")
    return model, X_train, X_test, y_train, y_test, y_pred


    