from single_variable_optimization import *
import numpy as np
import numdifftools as nd
import pandas as pd
import time
import matplotlib.pyplot as plt
import xlwt
from xlwt import Workbook

class Multi_optimization():
    def __init__(self, part, n, m, user_input, x):
        self.n = n
        self.part = part
        self.wb = Workbook()
        a,b = self.getrange()

        if user_input=="N":
            #self.x = np.random.uniform(low=a, high=b, size=(n))
            self.x = np.random.rand(n)
        else :
            self.x = np.array(x[0:n])

        print("Initial value for x : ", self.x)
        self.m = m

    def get_function_name(self):

        if self.part == 1:
            return "Sum Squares" 
        elif self.part == 2:
            return "Rosenbrock" 
        elif self.part == 3:
            return "Dixon Price"
        elif self.part == 4:
            return "Trid"
        elif self.part == 5:
            return "Zakharox"
        elif self.part == 6:
            return "Himmelblau"    

    def equation(self, x):
        eqn = 0
        if self.part == 1:
            for i in range(0, len(x)):
                eqn = eqn + (i+1)*x[i]*x[i]
        elif self.part == 2:
            for i in range(0, len(x)-1):
                # a = (x[i+1]-x[i]*x[i])
                # b = (x[i]-1)
                eqn = eqn + ((100*(x[i+1]-x[i]*x[i])**2)+((x[i]-1)**2))
        elif self.part == 3:
            eqn = (x[0]-1)*(x[0]-1)
            for i in range(1, len(x)):
                eqn = eqn + (i+1)*((2*x[i]*x[i]-x[i-1])**2)
        elif self.part == 4:
            eqn = (x[0]-1)*(x[0]-1)
            for i in range(1, len(x)):
                eqn = eqn + (x[i]-1)**2 - x[i]*x[i-1]
        elif self.part == 5:
            inter_val = 0
            for i in range(0, len(x)):
                inter_val = inter_val + (1/2)*(i+1)*x[i]

                eqn = eqn + x[i]*x[i]
            
            eqn = eqn + inter_val**2 + inter_val**4
        elif self.part == 6:
            eqn = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

        return eqn


    def gradient(self, x):
        epsilon = 2.34E-10

        grads = np.array([])
        f_x = self.equation(x)

        for i in range(self.n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] = x_plus[i]+epsilon
            x_minus[i] = x_minus[i]-epsilon

            grads = np.append(grads, ((self.equation(x_plus)-self.equation(x_minus))/(2*epsilon)))

        return grads

    def hessian(self, x):
        hess = np.eye(self.n, self.n)

        f_x = self.equation(x)
        epsilon = 10**-6

        for i in range(self.n):
            for j in range(i+1):
                x_plus = x.copy()
                x_minus = x.copy()
                if i==j:
                    x_plus[i] = x_plus[i]+epsilon
                    x_minus[i] = x_minus[i]-epsilon
                    hess[i][i] = (self.equation(x_plus)+self.equation(x_minus)-2*f_x)/(epsilon**2)
                else:
                    x_plus_i_plus_j = x.copy()
                    x_plus_i_plus_j[i] = x_plus_i_plus_j[i]+epsilon
                    x_plus_i_plus_j[j] = x_plus_i_plus_j[j]+epsilon

                    x_plus_i_minus_j = x.copy()
                    x_plus_i_minus_j[i] = x_plus_i_minus_j[i]+epsilon
                    x_plus_i_minus_j[j] = x_plus_i_minus_j[j]-epsilon

                    x_minus_i_plus_j = x.copy()
                    x_minus_i_plus_j[i] = x_minus_i_plus_j[i]-epsilon
                    x_minus_i_plus_j[j] = x_minus_i_plus_j[j]+epsilon

                    x_minus_i_minus_j = x.copy()
                    x_minus_i_minus_j[i] = x_minus_i_minus_j[i]-epsilon
                    x_minus_i_minus_j[j] = x_minus_i_minus_j[j]-epsilon

                    hess[i][j] = (self.equation(x_plus_i_plus_j)+self.equation(x_minus_i_minus_j)-self.equation(x_plus_i_minus_j)-self.equation(x_minus_i_plus_j))/(4*epsilon*epsilon)

                    hess[j][i] = hess[i][j]

        return hess

    def checkRestart(self, mat, f_grad, inverse):
        if np.all(np.linalg.eigvals(mat) >= 0):
            if np.matmul(np.matmul(f_grad, inverse), f_grad)<=0:
                return True
            else:
                return False
        else:
            return False


    def getrange(self):
        part = self.part
        n = self.n

        if part == 1:
            a = -5.12
            b = 5.12
        elif part == 2:
            a = -2.048
            b = 2.048
        elif part == 3:
            a = -10.0
            b = 10.0
        elif part == 4:
            a = -(n**2)
            b = n**2
        elif part== 5:
            a = -5.0
            b = 10.0
        elif part==6:
            a = -5.0
            b = 5.0
            
        return a, b

    def plot_f_x_versus_iterations(self):
        x = self.f_x_array
        print(f"f_x_array : {x}")
        k = self.k

        x_axis=np.array([])

        for i in range(1, k+1):
            x_axis = np.append(x_axis, i)

        y_axis = x[abs(len(x_axis)-len(x)):]

        fig= plt.figure()
        axes=fig.add_subplot(111)

        plt.title(f"Plot for Iterations vs Objective Function Value for dimension : {self.n}")

        plt.ylabel("F_X")
        plt.xlabel("Number of iterations")

        axes.plot(x_axis, y_axis)

        for i in range(0,len(y_axis)):
            plt.plot(x_axis[i], y_axis[i], 'ro')
        # plt.show(block=False)

        plt.savefig(f"./phase_2_graphs/iteration_plots/dim_{self.n}/question_{self.part}.png")

    def function_plot(self):
        a, b = self.getrange()
        x = np.linspace(-2,2,250)
        y = np.linspace(-2,2,250)

        X, Y = np.meshgrid(x, y)

        x_test = np.stack((X, Y), axis=0)
        Z = self.equation(x_test)

        #print(x_test)
        print(np.array(self.x_array)[:,0])

        fig = plt.figure(figsize = (16,8))

        iter_x = self.x_array[:,0]
        iter_y = self.x_array[:,0]

        anglesx = iter_x[1:] - iter_x[:-1]
        anglesy = iter_y[1:] - iter_y[:-1]

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
        ax.plot(iter_x,iter_y, self.equation(np.stack((iter_x, iter_y), axis=0)),color = 'r', marker = '*', alpha = .4)

        ax.view_init(45, 280)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax = fig.add_subplot(1, 2, 2)
        ax.contour(X,Y,Z, 50, cmap = 'jet')
        #Plotting the iterations and intermediate values
        ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
        ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
        ax.set_title('Gradient Descent with {} iterations'.format(self.k))

        plt.savefig(f"./phase_2_graphs/function_plots/question_{self.part}.png")


class Marquardt_method(Multi_optimization):
    def __init__(self, part, n, m, user_input, x):
        self.ld = 100
        self.epsilon = 10**-6
        #self.x = np.random.rand(n)
        super().__init__(part, n, m, user_input, x)
        

    def minimize(self):
        k = 0
        n = self.n
        x = self.x
        ld = self.ld
        func_eva = 0
        x_array = []
        f_x_array = []
        
        sheet1 = self.wb.add_sheet('Sheet 1')
        
        sheet1.write(0, 0, f"Function Name")
        sheet1.write(0, 1, f"{self.get_function_name()} Function")
        sheet1.write(1, 0, "Dimension")
        sheet1.write(1, 1, self.n)
        sheet1.write(2, 0, "Lower Bound")
        a, b = self.getrange()
        sheet1.write(2, 1, a)
        sheet1.write(3, 0, "Upper Bound")
        sheet1.write(3, 1, b)
        
        row_excel = 6
        while(True):
            sheet1.write(row_excel, 0, f"X_{k}")
            sheet1.write(row_excel, 1, np.array_str(x))
            row_excel+=1

            x_array.append(x)
            f_grad = self.gradient(x)
            func_eva += 2*n

            f_norm = np.linalg.norm(f_grad)

            if f_norm<=self.epsilon or k>=self.m:
                break
            
            f_x = self.equation(x)
            f_x_array.append(f_x)
            func_eva+=1

            while True:
                hessian_mat = self.hessian(x)
                func_eva += 2*n*n + 1

                print(f"n = {self.n}")
                iden = np.dot(ld, np.identity(self.n, dtype=float))

                inverse = np.linalg.inv(np.add(hessian_mat,iden))
                
                if self.checkRestart(np.add(hessian_mat,iden), f_grad, inverse):
                    print("RESTART CONDITION MET!!!")
                    time.sleep(2)
                    self.x = np.random.rand(n)
                    self.minimize()

                s_k = np.dot(-1, np.matmul(inverse, f_grad))

                bounding_phase_method = Bounding_phase_method(self.part, x, s_k)
                bounding_phase_method.minimize()
                a_bounding_phase, b_bounding_phase, func_eva_bounding_phase = bounding_phase_method.results()

                func_eva+=func_eva_bounding_phase
                print(f"--------------------------------------------------")
                print(f"Range from bounding phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")

                interval_halving_method = Interval_halving_method(self.part, x, s_k, a=a_bounding_phase, b=b_bounding_phase)
                interval_halving_method.minimize()
                a_interval_halving, b_interval_halving, func_eva_interval_halving = interval_halving_method.results()
                
                func_eva+= func_eva_interval_halving
                
                alpha = a_interval_halving + (b_interval_halving-a_interval_halving)/2

                print(f"--------------------------------------------------")
                print(f"Range from interval halving phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")

                x_plus_one = np.add(x, np.dot(alpha, s_k))

                
                #x_plus_one = np.add(x, s_k)

                func_eva+=1
                if self.equation(x_plus_one)<=f_x:
                    break

                ld = 2*ld

            ld = ld/2
            x = x_plus_one
            k=k+1
            
        
        print(f"Iterations used : {k}")
        print(f"Total Function evaluations : {func_eva}")
        self.k = k
        self.x = x
        self.func_eva = func_eva
        self.x_array = np.array(x_array)
        self.f_x_array = f_x_array

        row_excel += 3

        sheet1.write(row_excel, 0, "iterations")
        sheet1.write(row_excel, 1, k)
        row_excel+=1

        sheet1.write(row_excel, 0, "function evaluations")
        sheet1.write(row_excel, 1, func_eva)
        row_excel+=1

        sheet1.write(row_excel, 0, "Final Answer")
        sheet1.write(row_excel, 1, np.array_str(x))
        row_excel+=1

        self.wb.save(f"./phase_2_outputs/Marquardt_question_{self.part}_dim_{self.n}.xls")





    def results(self):
        return self.x, self.k, self.func_eva



def main():
    filename = input("Enter name of input file : ")
    df = pd.read_csv(f'./{filename}')
    xs = []
    itrs = []
    func_evas = []
    
    for i, row in df.iterrows():
        print(f"Input received from row {i}: {row}")
        time.sleep(1)
        part, n, m, user_input, x_string = row['part'], row['n'], row['m'], row['user_input'], row['x']

        if part>5:
            print("Value of part should be less than or equals to 5")
            continue

        print(f"--------------------------------------------------------------")
        x_array=[]

        for k in list(x_string.split(",")):
            x_array.append(float(k))

        marquardt = Marquardt_method(part, n, m, user_input, x_array)
        marquardt.minimize()

        print(f"--------------------------------------------------------------")
        x_result, itr, func_eva = marquardt.results()
        xs.append(x_result)
        itrs.append(itr)
        func_evas.append(func_eva)
        print(x_result)
        print(f"Results from marquardt method for part {int(part)}", x_result)
        print(f"Iterations for part {part} : {itr}")
        print(f"function evaluation for part {part} : {func_eva}")

        time.sleep(3)


if __name__ == "__main__":
    main()
    #rough()










