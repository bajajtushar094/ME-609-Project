import random
import math
from re import A, X
import numpy as np
import matplotlib.pyplot as plt
import calendar
import time

gmt = time.gmtime() 
ts = calendar.timegm(gmt)

def truncate_decimals(x):
    return x

class Basic_optimization():
    def __init__(self, part, x, x_k, s_k, r):
        self.part = part
        self.x_k = x_k
        self.s_k = s_k
        self.x = x
        self.r = r

    # def equation(self, x):
    #     if len(self.x_k)!=0 and len(self.s_k)!=0 : 
    #         x = np.add(self.x_k, np.dot(x, self.s_k))
    #     else:
    #         x = np.array([x])

    #     eqn = 0
    #     if self.part == 1:
    #         for i in range(0, len(x)):
    #             eqn = eqn + (i+1)*x[i]*x[i]
    #     elif self.part == 2:
    #         for i in range(0, len(x)-1):
    #             eqn = eqn + ((100*(x[i+1]-x[i]*x[i])**2)+((x[i]-1)**2))
    #     elif self.part == 3:
    #         eqn = (x[0]-1)*(x[0]-1)
    #         for i in range(1, len(x)):
    #             eqn = eqn + (i+1)*((2*x[i]*x[i]-x[i-1])**2)
    #     elif self.part == 4:
    #         eqn = (x[0]-1)*(x[0]-1)
    #         for i in range(1, len(x)):
    #             eqn = eqn + (x[i]-1)**2 - x[i]*x[i-1]
    #     elif self.part == 5:
    #         inter_val = 0
    #         for i in range(0, len(x)):
    #             inter_val = inter_val + (1/2)*(i+1)*x[i]

    #             eqn = eqn + x[i]*x[i]
            
    #         eqn = eqn + inter_val**2 + inter_val**4   
    #     elif self.part == 6:
    #         eqn = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

    #     return eqn

    def bracket_operator(self, x):
            if x<0:
                return x
            else:
                return 0

    def equation(self, x):
        if len(self.x_k)!=0 and len(self.s_k)!=0 : 
            x = np.add(self.x_k, np.dot(x, self.s_k))
        else:
            x = np.array([x])
            
        part = self.part
        r = self.r

        eqn = 0
        if part==1:
            eqn = (((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)+(r*(self.bracket_operator(((x[0]-5)**2)+(x[1]**2)-26))**2))
        elif part==2:
            eqn = ((x[0] - 10)**3+(x[1] - 20)**3)/7973 + r*(self.bracket_operator(((x[0]-5)**2 + (x[1]-5)**2)/100.0 - 1.0)**2) + r*(self.bracket_operator(-1*(((x[0] - 6)**2 + (x[1] - 5)**2)/82.81 - 1.0))**2)
        elif part ==3:
            eqn = -1*(((math.sin(2*math.pi*x[0])**3)*math.sin(2*math.pi*x[1]))/((x[0]**3)*(x[0]+x[1]))) + r*(self.bracket_operator(-1*(x[0]**2-x[1]+1)/101)**2) + r*(self.bracket_operator(-1*(1-x[0]+(x[1]-4)**2)/37)**2)
        elif part==4:
            eqn = (x[0]+x[1]+x[2])/30000 + r*(self.bracket_operator(-1*((-1+0.0025*(x[3]+x[5]))/4))**2) + r*(self.bracket_operator(-1*((-1+0.0025*(-x[3]+x[4]+x[6]))/4))**2) + r*(self.bracket_operator(-1*((-1+0.01*(-x[5]+x[7]))/9.0))**2) + r*(self.bracket_operator(-1*((100*x[0]-x[0]*x[5]+833.33252*x[3]-83333.333)/1650000))**2) + r*(self.bracket_operator(-1*((x[1]*x[3]-x[1]*x[6]-1250*x[3]+1250*x[4])/11137500))**2) + r*(self.bracket_operator(-1*((x[2]*x[4]-x[2]*x[7]-2500*x[4]+1250000)/11125000))**2)


        # print(f"eqn : {eqn}")
        return eqn



    def plot_range(self):
        x_axis = []
        y_axis = []

        a = int(self.a)
        b = int(self.b)

        new_a = self.new_a
        new_b = self.new_b

        for i in np.linspace(a,b,1000):
            y = self.equation(i)
            x_axis.append(i)
            y_axis.append(y)

        fig= plt.figure()
        axes=fig.add_subplot(111)
        if self.__class__.__name__=="Interval_halving_method":
            plt.title("Plot of Range for Interval Halving method")
        else :
            plt.title("Plot of Range for Bounding Phase method")

        plt.xlabel("x")
        plt.ylabel("f(x)")

        axes.plot(x_axis, y_axis)
        axes.plot([new_a], self.equation(new_a), 'ro')
        axes.plot([new_b], self.equation(new_b), 'ro')
        plt.show(block=False)
        plt.savefig(f"./graphs/range_plot_{self.__class__.__name__}_part{self.part}.png")




    def plot_x_versus_iterations(self):
        x = self.x
        k = self.k

        x_axis=np.array([])

        for i in range(0, k+1):
            x_axis = np.append(x_axis, i)

        y_axis = x[abs(len(x_axis)-len(x)):]

        fig= plt.figure()
        axes=fig.add_subplot(111)

        if self.__class__.__name__=="Interval_halving_method":
            plt.title("Iteration plot for Interval Halving method")
        else :
            plt.title("Iteration plot for Bounding Phase method")

        plt.ylabel("X")
        plt.xlabel("Number of iterations")

        axes.plot(x_axis, y_axis)

        for i in range(0,len(y_axis)):
            plt.plot(x_axis[i], y_axis[i], 'ro')
        plt.show(block=False)
        plt.savefig(f"./graphs/iterations_plot_{self.__class__.__name__}_part{self.part}.png")




class Bounding_phase_method(Basic_optimization):
    def __init__(self, part, x_k, s_k, r):
        x = np.array([random.random()])
        self.delta = 10**-3
        super().__init__(part, x, x_k, s_k, r)

    def minimize(self):
        out = open(f"./outputs/bounding_phase_method_part{self.part}.out", "w")
        k = 0
        x = self.x
        delta = self.delta
        
        f_x_minus_delta = super().equation(x[0]-delta)   
        f_x = super().equation(x[0])
        f_x_plus_delta = super().equation(x[0]+delta)

        function_eval= 3

        while True:
            if f_x_minus_delta >= f_x and f_x >= f_x_plus_delta :
                delta = abs(delta)
                break
            elif f_x_minus_delta <= f_x and f_x <= f_x_plus_delta :
                delta = -1 * abs(delta)
                break
            else :
                x = np.random.rand(1)
                delta = 10**-12

            f_x_minus_delta = super().equation(x[0]-delta)
            f_x = super().equation(x[0])
            f_x_plus_delta = super().equation(x[0]+delta)

            function_eval= function_eval+3
            break

        #out.write(f"Solving part with Bounding Phase Method- {self.part} \n a : {a} \n b : {b} \n delta : {delta} \n")

        out.write(f"#It\t\t\tx\t\t\tf_x\n")

        x = np.append(x, x[k] + ((2**k) * delta))

        #print("X :", x[k] + ((2**k) * delta))

        f_x_k_plus_one = super().equation(x[k+1])
        f_x_k = super().equation(x[k])

        
        while(f_x_k_plus_one<f_x_k): 
            
            out.write(f"{k}\t\t{truncate_decimals(x[k])}\t\t{truncate_decimals(f_x_k)}\n")
            #print(f"X value for {k}th iteration and x : {x[k]}")
            k=k+1
            x = np.append(x, x[k] + ((2**k) * delta))
            
            
            f_x_k = f_x_k_plus_one
            
            #Function Evaluation
            f_x_k_plus_one = super().equation(x[k+1])
            function_eval = function_eval+1

        out.write(f"{k}\t\t{truncate_decimals(x[k])}\t\t{truncate_decimals(f_x_k)}\n")
        #print(f"X value for {k}th iteration and x : {x[k]}")
        
        self.x = x
        self.delta = delta
        self.k = k
        self.new_a = self.x[self.k-1]
        self.new_b = self.x[self.k+1]
        self.func_eva = function_eval
        # print(f"Total function evaluation for bounding phase method : {function_eval}")

        out.write(f"Value for new a : {self.new_a} and new b : {self.new_b} after bounding phase method for {k} iterations")

        out.close()


    def results(self):
        return self.x[self.k-1], self.x[self.k+1], self.func_eva



class Interval_halving_method(Basic_optimization):
    def __init__(self, part, x_k, s_k, a, b, r):
        self.epsilon = 10**-3
        self.l = b-a
        self.a = a
        self.b = b
        x = np.array([])
        super().__init__(part, x, x_k, s_k, r)

    def minimize(self):
        out = open(f"./outputs/interval_halving_method_part{self.part}.out", "w")
        a = self.a
        b = self.b
        epsilon = self.epsilon
        l = self.l
        x_m = a + (b-a)/2
        x = self.x
        x = np.append(x, x_m)
        k=0
        out.write(f"Continue Solving part - {self.part} with Interval Halving Method \n a : {a} \n b : {b} \n")
        out.write(f"#It\t\t\ta\t\t\tb\t\t\tx_m\n")

        function_eval = 3
        
        while(abs(l)>epsilon):
            x_1 = a + l/4
            x_2 = b - l/4

            #Function Evaluation
            f_x_1 = super().equation(x_1)

            #Function Evaluation
            f_x_2 = super().equation(x_2)
            
            #Function Evaluation
            f_x_m = super().equation(x_m)


            if f_x_1 < f_x_2:
                b = x_m
                x_m = x_1
            else:
                if f_x_2 < f_x_m:
                    a = x_m
                    x_m = x_2
                else:
                    a = x_1
                    b = x_2

            l = b-a
            x = np.append(x, x_m)
            out.write(f"{k}\t\t\t{truncate_decimals(a)}\t\t\t{truncate_decimals(b)}\t\t\t{x_m}\n")
            #print(f"For {k}th iteration, A : {a}, B : {b} and X_M : {x_m}")
            k = k+1
            function_eval+=2
            break


        self.new_a = a
        self.new_b = b
        self.l = (b-a)
        self.k = k
        self.x = x
        self.func_eva = function_eval

        # print(f"Total function evaluation for interval halving method: {function_eval}")

        out.write(f"Value for new a : {self.new_a} and new b : {self.new_b} after interval halving method")

        out.close()

    def results(self):
        return self.new_a, self.new_b, self.func_eva