from multi_variable_optimization import *

def create_iteration_plots():

    for i in range(1, 6):
        itrs=[]
        func_evas = []
        for j in range(1, 11):
            print(f"--------------------------------------------------------------")
            marquardt = Marquardt_method(i, j, 100, "N", [5, 5, 5])
            marquardt.minimize()
            print(f"--------------------------------------------------------------")
            x, itr, func_eva = marquardt.results()
            itrs.append(itr)
            func_evas.append(func_eva)
            print(f"Results from marquardt method for part {i} and dimension {j} : {x}")
            print(f"Iterations for part {i} and dimension {j} : {type(itr)}")
            print(f"function evaluation for part {i} and dimension {j} : {func_eva}")
            marquardt.plot_f_x_versus_iterations()

        print(f"Iterations recovered {itrs}: ")

if __name__ == "__main__":
    create_iteration_plots()