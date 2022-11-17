from turtle import pen
from constraint_optimization import *

def create_iteration_plots():

    x_values = [[6,6],[6.1,6.1],[5.8,5.8],[6.4,6.4],[5.6,5.6],[6,6],[5.6,5.8],[6,6],[5.9,5.8],[6.05,6.05]]
    #x_values = [[20,33],[65,2],[99,55],[22.0,22.2],[52.03,1.66],[14.055,13.0],[14,0.3],[33.89,56.98],[100,0],[66,32]]
    out = open(f"./phase_3_outputs/question_2.out", "w")
    out.write(f'i\t\t\t\tf_x\t\t\t\tx\n')
    for i in range(10):
        penalty_function_method = Penalty_function_method(3, 2, i, 'N', x_values[i])

        penalty_function_method.minimize()
        x_answer, f_x = penalty_function_method.results()
        penalty_function_method.plot_p_x_versus_iterations()
        # penalty_function_method.function_plot()
        out.write(f'{i+1}\t\t\t\t{f_x}\t\t\t\t{x_values[i]}\n')

        print(f"final answer from penalty method -> x : {x_answer}")

    

        

if __name__ == "__main__":
    create_iteration_plots()