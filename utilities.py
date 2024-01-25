# load libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
from IPython.display import set_matplotlib_formats

print("utilities loaded")


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
            color='black',
            weight = 'bold') 