from multi_variable_optimization import *


def histogram(x_axis, y_axis, part, ylabel):
    plt.ylim(0, max(y_axis)+4)
    plt.xlabel("Dimension of input")
    plt.ylabel(f"Number of {ylabel}")
    plt.title(f"Plot of dimesion vs {ylabel} for question {part}")
    plt.bar(x= x_axis, height = y_axis, color='blue', width=0.3)

    for i in range(0,len(y_axis)):
        plt.plot(x_axis[i], y_axis[i], 'green')

    plt.plot(x_axis, y_axis, color='red')

    plt.savefig(f"./phase_2_graphs/bar_plots/{ylabel}/question_{part}.png")
    plt.figure().clear()


def create_histogram_plots():

    for i in range(1, 6):
        itrs=[]
        func_evas = []
        for j in range(1, 11):
            print(f"--------------------------------------------------------------")
            marquardt = Marquardt_method(i, j, 100,"N", [5, 5, 5])
            marquardt.minimize()
            print(f"--------------------------------------------------------------")
            x, itr, func_eva = marquardt.results()
            itrs.append(itr)
            func_evas.append(func_eva)
            print(f"Results from marquardt method for part {i} and dimension {j} : {x}")
            print(f"Iterations for part {i} and dimension {j} : {type(itr)}")
            print(f"function evaluation for part {i} and dimension {j} : {func_eva}")

        print(f"Iterations recovered {itrs}: ")
        histogram([1,2,3,4,5,6,7,8,9,10], itrs, i, "iterations")
        histogram([1,2,3,4,5,6,7,8,9,10], func_evas, i, "function evaluations")


if __name__ == "__main__":
    create_histogram_plots()