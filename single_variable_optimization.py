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
    return round(x, 4)

class Basic_optimization():
    def __init__(self, part, n, m, x_k=np.array([]), s_k=np.array([])):
        self.part = part
        self.n = n
        self.m = m
        self.x_k = x_k
        self.s_k = s_k

    def equation(self, x):
        if type(x) is not np.ndarray:
            x = np.add(self.x_k, np.dot(x, self.s_k))

        eqn = 0
        if self.part == 1:
            for i in range(0, len(x)):
                eqn = eqn + round((i+1)*x[i]*x[i], 4)
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
        if self.__class__.__name__=="interval_halving_method":
            plt.title("Plot of Range for Interval Halving method")
        else :
            plt.title("Plot of Range for Bounding Phase method")

        plt.xlabel("x")
        plt.ylabel("f(x)")

        axes.plot(x_axis, y_axis)
        axes.plot([new_a], [self.equation(new_a)], 'ro')
        axes.plot([new_b], [self.equation(new_b)], 'ro')
        plt.show(block=False)
        plt.savefig(f"./graphs/range_plot_{self.__class__.__name__}_part{self.part}.png")




    def plot_x_versus_iterations(self):
        x = self.x
        k = self.k

        x_axis=[]

        for i in range(0, k+1):
            x_axis.append(i)

        y_axis = x[abs(len(x_axis)-len(x)):]

        fig= plt.figure()
        axes=fig.add_subplot(111)

        if self.__class__.__name__=="interval_halving_method":
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
    def __init__(self, part, n, m, x_k, s_k):
        super().__init__(part, n, m, x_k, s_k)
        self.x = np.random.rand(1)
        # print("X values : ",self.x)
        self.delta = 10**-3

    def minimize(self):
        out = open(f"./outputs/bounding_phase_method_part{self.part}.out", "w")
        k = 0
        # a = self.a
        # b = self.b
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

        #out.write(f"Solving part with Bounding Phase Method- {self.part} \n a : {a} \n b : {b} \n delta : {delta} \n")

        out.write(f"#It\t\t\tx\t\t\tf_x\n")

        x = np.append(x, x[k] + ((2**k) * delta))

        print("X :", x[k] + ((2**k) * delta))

        f_x_k_plus_one = super().equation(x[k+1])
        f_x_k = super().equation(x[k])

        
        while(f_x_k_plus_one<=f_x_k): 
            
            out.write(f"{k}\t\t{truncate_decimals(x[k])}\t\t{truncate_decimals(f_x_k)}\n")
            print(f"X value for {k}th iteration and x : {x[k]}")
            k=k+1
            x = np.append(x, x[k] + ((2**k) * delta))
            
            
            f_x_k = f_x_k_plus_one
            
            #Function Evaluation
            f_x_k_plus_one = super().equation(x[k+1])
            function_eval = function_eval+1

        out.write(f"{k}\t\t{truncate_decimals(x[k])}\t\t{truncate_decimals(f_x_k)}\n")
        print(f"X value for {k}th iteration and x : {x[k]}")
        
        self.x = x
        self.delta = delta
        self.k = k
        self.new_a = self.x[self.k-1]
        self.new_b = self.x[self.k+1]
        print(f"Total function evaluation for bounding phase method : {function_eval}")

        out.write(f"Value for new a : {self.new_a} and new b : {self.new_b} after bounding phase method for {k} iterations")

        out.close()


    def results(self):
        return self.x[self.k-1], self.x[self.k+1]



class Interval_halving_method(Basic_optimization):
    def __init__(self, a , b, part, n, m, x_k, s_k):
        super().__init__(part, n, m, x_k, s_k)
        self.epsilon = 10**-3
        self.l = b-a
        self.x = np.array([])
        self.a = a
        self.b = b

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
            print(f"For {k}th iteration, A : {a}, B : {b} and X_M : {x_m}")
            k = k+1
            function_eval+=2


        self.new_a = a
        self.new_b = b
        self.l = (b-a)
        self.k = k
        self.x = x

        print(f"Total function evaluation for interval halving method: {function_eval}")

        out.write(f"Value for new a : {self.new_a} and new b : {self.new_b} after interval halving method")

        out.close()

    def results(self):
        return self.new_a, self.new_b


def main():
    part = int(input("Enter a number between 1 and 6 to solve correspinding part of question: "))

    a = float(input("Enter a : "))

    b = float(input("Enter b : "))

    if part>6:
        print("Please enter correct part to be solved!")
        return 0

    maxi = input("Wether the function has to be maximize or not. Enter Y/N: ")

    maximize = False
    if maxi=="Y":
        maximize = True

    print(f"\n\n--------------------------------------------------\n\n")

    bounding_phase = Bounding_phase_method(a, b, maximize, part)
    bounding_phase.minimize()
    a_bounding_phase, b_bounding_phase = bounding_phase.results()
    print(f"--------------------------------------------------")
    print(f"Range from bounding phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")
    bounding_phase.plot_range()
    bounding_phase.plot_x_versus_iterations()

    print(f"\n\n--------------------------------------------------\n\n")

    interval_halving = Interval_halving_method(a_bounding_phase, b_bounding_phase, maximize, part)
    interval_halving.minimize()
    a_interval_halving, b_interval_halving = interval_halving.results()
    print(f"--------------------------------------------------")
    print(f"Range from interval halving phase method => a : {a_interval_halving}, b : {b_interval_halving}")
    interval_halving.plot_range()
    interval_halving.plot_x_versus_iterations()

    print(f"\n\n--------------------------------------------------\n\n")

    plt.show()

if __name__ == "__main__":
    main()