from single_variable_optimization import *
import numpy as np
import numdifftools as nd
import pandas as pd
import time
import matplotlib.pyplot as plt

def create_bar_plot(ylabel, part):
    x1 = [-10, 1, -1, 2, -2]
    x2 = [-10,1,-1,2,-2] 
    x3 = [-10, 1, -1, 2, -2] 
    x4 = [-10,1,-1,2,-2]
    x5 = [-10,1,-1,2,-2]
    
    itr = [1, 1, 1, 20, 25]
    func_eva = [147, 144, 138]


    # set width of bar
    barWidth = 0.15
    fig = plt.subplots(figsize =(15, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(x1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    
    # Make the plot
    plt.bar(br1, x1, width = barWidth, label ='x1', color='cornflowerblue')
    plt.bar(br2, x2, width = barWidth, label ='x2', color='lightsteelblue')
    plt.bar(br3, x3, width = barWidth, label ='x3', color='cornflowerblue')
    plt.bar(br4, x4, width = barWidth, label='x4', color='lightsteelblue')
    plt.bar(br5, x5, width = barWidth, label='x5', color='cornflowerblue')
    
    # Adding Xticks
    plt.xlabel(f'number of {ylabel}', fontweight ='bold', fontsize = 15)
    plt.ylabel('x_i', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(x1))],
            [1, 10, 15, 20, 25])
    
    plt.legend()
    plt.show() 
    plt.savefig(f"./phase_2_graphs/bar_plots/{ylabel}/question_{part}.png")

create_bar_plot('iterations', 1)