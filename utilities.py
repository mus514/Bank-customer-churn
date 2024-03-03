# load libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
from IPython.display import set_matplotlib_formats
from scipy.stats import ttest_ind
import scipy.stats as ss
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV


#--------------------------------------------
# Bar function
#--------------------------------------------
def bar_plot(x, x_label = None, y_label = None, title = None, color = None):
    """
    Create a bar plot for categorical data.

    Parameters:
    - x: Pandas Series or DataFrame column
        The categorical data to be plotted.
    - x_label: str, optional
        Label for the x-axis. Default is None.
    - y_label: str, optional
        Label for the y-axis. Default is None.
    - title: str, optional
        Title for the plot. Default is None.
    - color: str or list of str, optional
        Color(s) for the bars. If a single color is provided, all bars will have the same color.
        If a list is provided, each bar will be colored accordingly. Default is None.
    
    Returns:
        None
    """
     # Calculate the percentage and plot the bar chart
    obj = (x.value_counts(normalize=True)*100).plot(kind = 'bar', color = color)

     # Set labels and title
    obj.set_ylabel(y_label)
    obj.set_xlabel(x_label)
    obj.set_title(title)

    # Customize x-axis ticks
    x_axis_name = list(x.value_counts(normalize=True).keys())
    plt.xticks(list(range(0, len(x_axis_name))), x_axis_name, rotation=0)

    # Add percentage labels on top of each bar
    for i in obj.patches:
        obj.text(i.get_x()+.10, i.get_height()-4.0,\
            f'{round((i.get_height()), 2)}%',
            fontsize=10,
            color='white',
            weight = 'bold')
        



#--------------------------------------------
# Student function
#--------------------------------------------
def student_test(df, numeric_col, group_by, alpha_level):
    """
    Perform independent two-sample t-tests for each numeric column in a DataFrame based on a grouping variable.

    Parameters:
    - df: pandas DataFrame
        The DataFrame containing the data.
    - numeric_col: list of str
        The list of numeric column names to be tested.
    - group_by: str
        The column name used for grouping the data.
    - alpha_level: float
        The significance level for hypothesis testing.

    Returns:
    None
    """
    for key in numeric_col:
        # Extract data for each group
        group = df[group_by].unique()
        group_1 = df[df[group_by] == group[0]][key]
        group_2 = df[df[group_by] == group[1]][key]

        # Perform two-sample t-test
        t_statistic, p_value = ttest_ind(group_1, group_2)

        # Check for significance and print results
        if p_value < alpha_level:
            print("[{}, {}] : Reject the null hypothesis with level {}; there is a significant difference between groups.".
                  format(key, group_by, alpha_level))
        else:
            print('[{}, {}] : Fail to reject the null hypothesis with level {}; there is no significant difference between groups.'.
                  format(key, group_by, alpha_level))  
        print('===')


#--------------------------------------------
# Cramer V function
#--------------------------------------------
def CramerV(confusion_matrix, bias = False):
    """
    Calculate Cramer's V statistic for categorical variable association.

    Parameters:
    - confusion_matrix: 2D array-like
        The confusion matrix of categorical variables.
    - bias: bool, optional
        If True, applies a bias correction to the chi-square statistic. Default is False.

    Returns:
    float
        Cramer's V statistic.
    """
    chi = ss.chi2_contingency(confusion_matrix)[0]
    r, k = confusion_matrix.shape
    n = confusion_matrix.sum()

    if bias:
        chi = max(0, chi-((k-1)*(r-1))/(n-1))
        k = k-((k-1)**2)/(n-1)
        r = r-((r-1)**2)/(n-1)
        v = np.sqrt(chi/n*min(r-1, k-1)) 

    else:
        v = np.sqrt((chi/n)/min(r-1, k-1)) 
    
    return round(v, 2)



#--------------------------------------------
# Best cut probability 
#--------------------------------------------
def best_cut_prob(pred, y, cut_prob = np.arange(0, 1.01, 0.01)):
    """
    Find the best classification probability threshold based on predicted probabilities.

    Parameters:
    - pred: array-like
        Predicted probabilities for binary classification.
    - y: array-like
        True binary labels.
    - cut_prob: array-like, optional
        Array of probability thresholds to test. Default is np.arange(0, 1.01, 0.01).

    Returns:
    tuple
        Tuple containing the best classification accuracy, the corresponding best probability threshold,
        and a list of classification accuracies for each tested threshold.
    """
    result = []

    # Iterate over each probability threshold
    for p in cut_prob:
        # Convert predicted probabilities to binary predictions using the threshold
        y_hat = (pred > p)*1
        # Calculate classification accuracy for the current threshold
        score = ((y==0)*(y_hat==0) + (y==1)*(y_hat==1)).mean()
        result.append(score)

    # Find the maximum accuracy and the corresponding threshold
    best_score = np.max(result)
    best_prob = cut_prob[result.index(best_score)]

    return best_score, best_prob, result



#--------------------------------------------
# CV to calculat the train probabilities
#--------------------------------------------
def cv_pred_y(X_train, y_train, model, k=10):
    """
    Perform cross-validated predictions using a given model.

    Parameters:
    - X_train: array-like
        The training data.
    - y_train: array-like
        The corresponding true labels.
    - model: sklearn estimator
        The machine learning model to use for predictions.
    - k: int, optional
        Number of folds for cross-validation. Default is 10.

    Returns:
    array
        Array of cross-validated predicted probabilities.
    """
    cv = KFold(k)
    y_hat = np.zeros(X_train.shape[0]) 

    # Iterate through each fold
    for i, j in cv.split(X_train):
        X_train_temp = X_train[i, :]
        y_train_temp = y_train[i]
        X_test_temp = X_train[j, :]

        # Perform cross-validated training and get predicted probabilities
        cross_val_scores = cross_val_score(model, X_train_temp, y_train_temp, cv=5, scoring='f1')
        model.fit(X_train_temp, y_train_temp)
        y_hat[j] = model.predict_proba(X_test_temp)[:,1]
    
    return y_hat




print("utilities loaded")