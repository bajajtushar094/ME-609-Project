from venv import create
from multi_variable_optimization import *

def create_surface_plots():

    for i in range(1,6):
        marquardt = Marquardt_method(i, 2, 100, "N", [36,36])
        marquardt.minimize()

        marquardt.function_plot()

if __name__ == "__main__":
    create_surface_plots()