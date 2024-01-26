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


# Bar plot function
def bar_plot(x, x_label = None, y_label = None, title = None, color = None):
    obj = (x.value_counts(normalize=True)*100).plot(kind = 'bar', color = color)
    obj.set_ylabel(y_label)
    obj.set_xlabel(x_label)
    obj.set_title(title)
    x_axis_name = list(x.value_counts(normalize=True).keys())
    plt.xticks(list(range(0, len(x_axis_name))), x_axis_name, rotation=0)
    for i in obj.patches:
        obj.text(i.get_x()+.10, i.get_height()-4.0,\
            f'{round((i.get_height()), 2)}%',
            fontsize=10,
            color='white',
            weight = 'bold')

# Student test function
def student_test(df, numeric_col, group_by, alpha_level):
    for key in numeric_col:
        group = df[group_by].unique()
        group_1 = df[df[group_by] == group[0]][key]
        group_2 = df[df[group_by] == group[1]][key]
        t_statistic, p_value = ttest_ind(group_1, group_2)
        if p_value < alpha_level:
            print("[{}, {}] : Reject the null hypothesis with level {}; there is a significant difference between groups.".
                  format(key, group_by, alpha_level))
        else:
            print('[{}, {}] : Fail to reject the null hypothesis with level {}; there is no significant difference between groups.'.
                  format(key, group_by, alpha_level))  
        print('===')


print("utilities loaded")