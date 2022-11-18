from single_variable_optimization import *
import numpy as np
import numdifftools as nd
import pandas as pd
import time
import matplotlib.pyplot as plt
import xlwt
from xlwt import Workbook

# Base Class for Multi Variate Optimization
class Multi_optimization():

    #Constructor for Base Class
    def __init__(self, part, n, m, user_input, x, r):
        self.n = n
        self.part = part
        self.wb = Workbook()
        a,b = self.getrange()

        if user_input=="N":
            # self.x = np.random.uniform(low=a, high=b, size=(n))
            self.x = np.array(x)
        else :
            self.x = np.array(x)

        print("Initial value for x : ", self.x)
        self.m = m
        self.r = r

    def bracket_operator(self, x):
            if x<0:
                return x
            else:
                return 0


    #Get Function Name corresponding to part entered
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
        part = self.part
        r = self.r

        eqn = 0
        if part==1:
            eqn = (((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)+(r*(self.bracket_operator(((x[0]-5)**2)+(x[1]**2)-26))**2) + r*(self.bracket_operator(x[0])**2) + r*(self.bracket_operator(x[1])**2))
        elif part==2:
            eqn = ((x[0] - 10)**3+(x[1] - 20)**3)/7973 + r*(self.bracket_operator(((x[0]-5)**2 + (x[1]-5)**2)/100.0 - 1.0)**2) + r*(self.bracket_operator(-1*(((x[0] - 6)**2 + (x[1] - 5)**2)/82.81 - 1.0))**2)
        elif part ==3:
            eqn = -1*(((math.sin(2*math.pi*x[0])**3)*math.sin(2*math.pi*x[1]))/((x[0]**3)*(x[0]+x[1]))) + r*(self.bracket_operator(-1*(x[0]**2-x[1]+1)/101)**2) + r*(self.bracket_operator(-1*(1-x[0]+(x[1]-4)**2)/37)**2) + r*(self.bracket_operator(x[0]/10)**2) + r*(self.bracket_operator(x[1]/10)**2) + r*(self.bracket_operator((10-x[0])/10)**2) + r*(self.bracket_operator((10-x[1])/10)**2) 
        elif part==4:
            eqn = (x[0]+x[1]+x[2])/30000 + r*(self.bracket_operator(-1*((-1+0.0025*(x[3]+x[5]))/4))**2) + r*(self.bracket_operator(-1*((-1+0.0025*(-x[3]+x[4]+x[6]))/4))**2) + r*(self.bracket_operator(-1*((-1+0.01*(-x[5]+x[7]))/9.0))**2) + r*(self.bracket_operator(-1*((100*x[0]-x[0]*x[5]+833.33252*x[3]-83333.333)/1650000))**2) + r*(self.bracket_operator(-1*((x[1]*x[3]-x[1]*x[6]-1250*x[3]+1250*x[4])/11137500))**2) + r*(self.bracket_operator(-1*((x[2]*x[4]-x[2]*x[7]-2500*x[4]+1250000)/11125000))**2)
            eqn += r*(self.bracket_operator((x[0]-100)/10000)**2) + r*(self.bracket_operator((10000-x[0])/10000)**2) 
            
            eqn += r*(self.bracket_operator((x[1]-1000)/10000)**2) + r*(self.bracket_operator((10000-x[1])/10000)**2) 
            eqn += r*(self.bracket_operator((x[2]-1000)/10000)**2) + r*(self.bracket_operator((10000-x[2])/10000)**2)

            eqn += r*(self.bracket_operator((x[3]-10)/1000)**2) + r*(self.bracket_operator((1000-x[3])/1000)**2)
            eqn += r*(self.bracket_operator((x[4]-10)/1000)**2) + r*(self.bracket_operator((1000-x[4])/1000)**2)
            eqn += r*(self.bracket_operator((x[5]-10)/1000)**2) + r*(self.bracket_operator((1000-x[5])/1000)**2)
            eqn += r*(self.bracket_operator((x[6]-10)/1000)**2) + r*(self.bracket_operator((1000-x[6])/1000)**2)
            eqn += r*(self.bracket_operator((x[7]-10)/1000)**2) + r*(self.bracket_operator((1000-x[7])/1000)**2)


        # print(f"eqn : {eqn}")
        return eqn

    #function for calculating gradient
    def gradient(self, x):
        return  nd.Gradient(self.equation)(x)

    #function for calculating Hessian matrix
    def hessian(self, x):
        Hessian_func = nd.Hessian(self.equation)

        hessian = Hessian_func(x)

        return hessian

    #function to check restart condition
    def checkRestart(self, mat, f_grad, inverse):
        if np.all(np.linalg.eigvals(mat) >= 0):
            if np.matmul(np.matmul(f_grad, inverse), f_grad)<=0:
                return True
            else:
                return False
        else:
            return False

    #function to get range of the question
    def getrange(self):
        part = self.part
        n = self.n

        if part == 1:
            a = -100.0
            b = 100.0
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

    #function for plotting f_x vs iteration graph
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

    #function to plot Surface and Countor plots of the question
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


#Class for Marquardt Method: it extends Multi_optimization class
class Marquardt_method(Multi_optimization):

    #Constructor for Marquardt Method
    def __init__(self, part, n, m, user_input, x, r):
        self.ld = 100
        self.epsilon = 10**-3
        super().__init__(part, n, m, user_input, x, r)
        
    #Marquart Method
    def minimize(self):
        k = 0
        n = self.n
        x = self.x
        ld = self.ld
        func_eva = 0
        x_array = []
        f_x_array = []
        r = self.r
        
        #Store output in the excel file
        sheet1 = self.wb.add_sheet('Sheet 1')
        
        #Entries for excel file
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

            #Calculate gradient at x
            f_grad = self.gradient(x)

            #Function evaluation for Gradient
            func_eva += 2*n

            #Calculate norm of Gradient at x    
            f_norm = np.linalg.norm(f_grad)

            # print(f"F_Norm : {f_norm}")

            #Break the outer loop if Norm of Gradient at x is less than epsilon or Iteration have increased beyond M
            if f_norm<=self.epsilon or k>=self.m:
                break
            
            f_x = self.equation(x)
            f_x_array.append(f_x)
            func_eva+=1

            while True:
                #Calculate Hessian Matrix
                hessian_mat = self.hessian(x)
                #Function evaluation for Hessian Matrix is (2*N^2+1)
                func_eva += 2*n*n + 1

                #lambda * Indentity Matrix
                iden = np.dot(ld, np.identity(self.n, dtype=float))

                #Calculating the inverse
                inverse = np.linalg.inv(np.add(hessian_mat,iden))
                
                #Check for restart condition
                if self.checkRestart(np.add(hessian_mat,iden), f_grad, inverse):
                    print("RESTART CONDITION MET!!!")
                    time.sleep(2)
                    self.x = np.random.rand(n)
                    self.minimize()

                #Calculate Direction of descent at x
                s_k = np.dot(-1, np.matmul(inverse, f_grad))

                #Optimize to get value of alpha
                bounding_phase_method = Bounding_phase_method(self.part, x, s_k, self.r)
                bounding_phase_method.minimize()
                a_bounding_phase, b_bounding_phase, func_eva_bounding_phase = bounding_phase_method.results()

                #Add function evaluations from bounding phase method
                func_eva+=func_eva_bounding_phase

                #Call Interval Halving Method by giving the results of Bounding Phase Method
                interval_halving_method = Interval_halving_method(self.part, x, s_k, a_bounding_phase, b_bounding_phase, r)
                interval_halving_method.minimize()
                a_interval_halving, b_interval_halving, func_eva_interval_halving = interval_halving_method.results()
                
                #Add function evaluations from Interval Halving Method
                func_eva+= func_eva_interval_halving
                
                #Use average value of Lower and Upper bounds from Interval Halving Method
                alpha = a_interval_halving + (b_interval_halving-a_interval_halving)/2

                #Calculate value of X_k+1
                x_plus_one = np.add(x, s_k)

                #Increase Function Evaluation for calculating function value at x_k+1
                func_eva+=1
                #Break the loop if the value of f_x_k+1 decreased
                if self.equation(x_plus_one)<=f_x:
                    break
                
                #Increase lambda if value of f_x_k+1 increased
                ld = 2*ld

            #Decrease the value of lambda if value of function at x_k+1 decreased
            ld = ld/2

            #Assign x_k+1 to x_k 
            x = x_plus_one

            #Increase the iteration counter
            k=k+1
            
        
        print(f"Iterations used : {k}")
        print(f"Total Function evaluations : {func_eva}")
        self.k = k
        self.x = x
        self.func_eva = func_eva
        self.x_array = np.array(x_array)
        self.f_x_array = f_x_array

        row_excel += 3

        #Store final answer in excel sheet
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




    #Function to get x, iteration counter and function evaluation for Marquardt Method
    def results(self):
        return self.x, self.k, self.func_eva



def main():
    #Get Filename from user
    filename = input("Enter name of input file : ")

    #Read Particular CSV file as dataframe
    df = pd.read_csv(f'./{filename}')
    
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

        #Instantiate Object for Marquardt Method class
        marquardt = Marquardt_method(part, n, m, user_input, x_array, 1)
        marquardt.minimize()

        print(f"--------------------------------------------------------------")
        x_result, itr, func_eva = marquardt.results()
        print(x_result)
        print(f"Results from marquardt method for part {int(part)}", x_result)
        print(f"Iterations for part {part} : {itr}")
        print(f"function evaluation for part {part} : {func_eva}")

        time.sleep(3)


if __name__ == "__main__":
    main()










