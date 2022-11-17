from multi_variable_optimization import *
import math
from matplotlib.ticker import FormatStrFormatter

class Constraint_optimization():
    def __init__(self, part, n, input, user_input, x):
        self.part = part
        self.n = n
        self.user_input = user_input
        self.input = input
        self.x = np.array(x)
        self.c = 2
        self.r = 0.05

    def bracket_operator(self, x):
        if x<0:
            return x
        else:
            return 0


    def equation(self, x, r):
        part = self.part

        eqn = 0
        if part==1:
            eqn = (((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)+(r*(self.bracket_operator(((x[0]-5)**2)+(x[1]**2)-26))**2))
        elif part==2:
            eqn = ((x[0] - 10)**3+(x[1] - 20)**3)/7973 + r*(self.bracket_operator(((x[0]-5)**2 + (x[1]-5)**2)/100.0 - 1.0)**2) + r*(self.bracket_operator(-1*(((x[0] - 6)**2 + (x[1] - 5)**2)/82.81 - 1.0))**2)
        elif part ==3:
            eqn = -1*(((math.sin(2*math.pi*x[0])**3)*math.sin(2*math.pi*x[1]))/((x[0]**3)*(x[0]+x[1]))) + r*(self.bracket_operator(-1*(x[0]**2-x[1]+1)/101)**2) + r*(self.bracket_operator(-1*(1-x[0]+(x[1]-4)**2)/37)**2)
        elif part==4:
            eqn = (x[0]+x[1]+x[2])/30000 + r*(self.bracket_operator(-1*((-1+0.0025*(x[3]+x[5]))/4))**2) + r*(self.bracket_operator(-1*((-1+0.0025*(-x[3]+x[4]+x[6]))/4))**2) + r*(self.bracket_operator(-1*((-1+0.01*(-x[5]+x[7]))/9.0))**2) + r*(self.bracket_operator(-1*((100*x[0]-x[0]*x[5]+833.33252*x[3]-83333.333)/1650000))**2) + r*(self.bracket_operator(-1*((x[1]*x[3]-x[1]*x[6]-1250*x[3]+1250*x[4])/11137500))**2) + r*(self.bracket_operator(-1*((x[2]*x[4]-x[2]*x[7]-2500*x[4]+1250000)/11125000))**2)
    
        return eqn

    def func_equation(self, x):
        part = self.part

        eqn = 0
        if part==1:
            eqn = (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
        elif part==2:
            eqn = (x[0] - 10)**3+(x[1] - 20)**3
        elif part ==3:
            eqn = -1*(((math.sin(2*math.pi*x[0])**3)*math.sin(2*math.pi*x[1]))/((x[0]**3)*(x[0]+x[1])))
        elif part==4:
            eqn = x[0]+x[1]+x[2]
    
        return eqn


    def plot_p_x_versus_iterations(self):
        x = self.f_x_array
        print(f"f_x_array : {x}")
        k = self.k

        x_axis=np.array([])

        for i in range(1, k+1):
            x_axis = np.append(x_axis, i)

        y_axis = x[abs(len(x_axis)-len(x)):]

        fig= plt.figure()
        axes=fig.add_subplot(111)

        plt.title(f"Plot for Iterations vs Objective Function Value for input : {self.input+1}")

        plt.ylabel("F_X")
        plt.xlabel("Number of iterations")

        axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes.plot(x_axis, y_axis)

        for i in range(0,len(y_axis)):
            plt.plot(x_axis[i], y_axis[i], 'ro')
        # plt.show(block=False)

        plt.savefig(f"./phase_3_graphs/iteration_plots/question_{self.part}/input_{self.input+1}.png")

    #function to plot Surface and Countor plots of the question
    def function_plot(self):
        x = np.linspace(-10,10,250)
        y = np.linspace(-10,10,250)

        X, Y = np.meshgrid(x, y)

        x_test = np.stack((X, Y), axis=0)
        Z = self.func_equation(x_test)

        # print(self.x_array)
        # print(np.array(self.x_array)[:,0])

        fig = plt.figure(figsize = (16,8))

        iter_x = self.x_array[:,0]
        iter_y = self.x_array[:,0]

        anglesx = iter_x[1:] - iter_x[:-1]
        anglesy = iter_y[1:] - iter_y[:-1]

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
        ax.plot(iter_x,iter_y, self.func_equation(np.stack((iter_x, iter_y), axis=0)),color = 'r', marker = '*', alpha = .4)

        ax.view_init(45, 280)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(1, 2, 2)
        ax.contour(X,Y,Z, 50, cmap = 'jet')
        #Plotting the iterations and intermediate values
        ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
        ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
        ax.set_title(f'Bracket Penalty Function Method for question {self.part-1} and input {self.input+1}')

        plt.savefig(f"./phase_3_graphs/function_plots/question_{self.part}/input_{self.input+1}.png")


class Penalty_function_method(Constraint_optimization):
    def __init__(self, part, n, input, user_input, x):
        print(f"received x value : {type(x)}")
        super().__init__(part, n, input, user_input, x)

    def minimize(self):
        x = self.x
        part = self.part
        n = self.n
        r = self.r
        c = self.c
        r_array = [r]
        func_evaluations = 0
        f_x_array = []
        x_array = [x]

        k=1

        marquardt_method = Marquardt_method(part, n, 1000, 'N', x, r)
        marquardt_method.minimize()

        x_plus_one, itrs, func_eval = marquardt_method.results()
        print(f"x_1 : {x_plus_one}")
        print(f"r : {c*r}")
        r_array.append(c*r)
        f = self.func_equation(x)
        f_x_array.append(f)
        f_plus_one = self.func_equation(x_plus_one)
        f_x_array.append(f_plus_one)
        func_evaluations += func_eval
        x = x_plus_one
        x_array.append(x)

        while True:
            
            marquardt_method = Marquardt_method(part, n, 1000, 'N', x, r_array[-1])
            marquardt_method.minimize()

            x_plus_one, itrs, func_eval = marquardt_method.results()
            func_evaluations += func_eval

            p = self.equation(x, r_array[-1])
            p_plus_one = self.equation(x_plus_one, r_array[-2])
            
            f = self.func_equation(x)
            f_x_array.append(f)
            f_plus_one = self.func_equation(x_plus_one)
            f_x_array.append(f_plus_one)

            print(f"\n\n\n----------------------------\n\n\n")

            print(f"x_0 after {k} : {x}")
            print(f"x_1 after {k} : {x_plus_one}")
            print(f"p_0 after {k} : {p}")
            print(f"p_1 after {k} : {p_plus_one}")
            print(f"Constraint Violation : {abs(p_plus_one-p)}")
            print(f"\n\n\n----------------------------\n\n\n")

            
            if abs(p_plus_one-p)<10**-3:
                break

            r_array.append(c*r_array[-1])
            x = x_plus_one
            x_array.append(x)
            k+=1

        self.x = x_plus_one
        x_array.append(x_plus_one)
        self.f_x_array = f_x_array
        self.func_evaluations = func_evaluations
        self.k = k
        self.x_array = np.array(x_array)
    

    def results(self):
        return self.x, self.f_x_array[-1]


def main():
    #Get Filename from user
    filename = input("Enter name of input file : ")

    df = pd.read_csv(f'./{filename}')
    #Read Particular CSV file as dataframe
    for i, row in df.iterrows(): 
        print(f"Input received from row {i}: {row}")
        time.sleep(1)
        part, n, user_input, x_string = row['part'], row['n'], row['user_input'], row['x']

        print(f"--------------------------------------------------------------")
        x_array=[]

        for k in list(x_string.split(",")):
            x_array.append(float(k))


        penalty_function_method = Penalty_function_method(part, n, 1, user_input, [20,33])

        penalty_function_method.minimize()
        x = penalty_function_method.results()

        print(f"final answer from penalty method -> x : {x}")

        penalty_function_method.plot_p_x_versus_iterations()

if __name__ == "__main__":
    main()
            