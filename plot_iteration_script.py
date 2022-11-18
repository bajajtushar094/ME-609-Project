from turtle import pen
from constraint_optimization import *

def create_iteration_plots():

    # x_values = [[500,500,500,500,500,500,500,500],[2,2,2,2,2,2,2,2],[120,120,120,120,120,120,120,120],[700,700,700,700,700,700,700,700],[1000,1000,1000,1000,1000,1000,1000,1000],[400,400,400,400,400,400,400,400],[20,20,20,20,20,20,20,20],[120,120,120,120,120,120,120,120],[250,250,250,250,250,250,250,250],[380,380,380,380,380,380,380,380]]
    #x_values = [[20,33],[65,2],[99,55],[22.0,22.2],[52.03,1.66],[14.055,13.0],[14,0.3],[33.89,56.98],[100,0],[66,32]]
    #x_values = [[1.1,1.1],[6.1,.5],[6.2,0.1],[5.1,3.1],[0.5,0.5],[1.0,7.5],[2.1,2.9],[1.5,2.06],[1.05,8.95],[1.0001,1.9999]]
    x_values = [[1.1,1.1],[2.1,2.1],[0.2,0.2],[2.9,2.9],[3.9,3.9],[4.5,4.5],[1.5,1.5],[1.9,1.9],[3.8,3.8],[1.4,1.4],[4.9,4.9]]
    out = open(f"./phase_3_outputs/himmelblau.out", "w")
    out.write(f'i\t\t\t\tf_x\t\t\t\tx\n')
    for i in range(10):
        penalty_function_method = Penalty_function_method(1, 2, i, 'N', x_values[i])

        penalty_function_method.minimize()
        x_answer, f_x = penalty_function_method.results()
        penalty_function_method.plot_p_x_versus_iterations()
        # penalty_function_method.function_plot()
        out.write(f'{i+1}\t\t\t\t{f_x}\t\t\t\t{x_values[i]}\n')

        print(f"final answer from penalty method -> x : {x_answer}")

    

        

if __name__ == "__main__":
    create_iteration_plots()