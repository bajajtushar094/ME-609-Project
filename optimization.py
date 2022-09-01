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

class basic_optimization():
    def __init__(self, a, b, maximize, part):
        self.a = a
        self.b = b
        self.maximize = maximize
        self.part = part

    def equation(self, x):  
        if self.part==1:
            eqn = (2*x-5)**4 - (x**2-1)**3
        elif self.part==2:
            eqn = 8 + x**3 - 2*x - 2 * math.exp(x)
        elif self.part==3:
            eqn = 4 * x * math.sin(x)
        elif self.part==4:
            eqn = 2 * (x-3)**2 + math.exp(0.5 * x**2)
        elif self.part==5:
            eqn = x**2 - 10*math.exp(0.1*x)
        elif self.part==6:
            eqn = 20*math.sin(x) -15 * x**2
        elif self.part==7:
            eqn = x**2 + 54/x

        
        if self.maximize==True:
            eqn = -1*eqn

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

        axes.plot(x_axis, y_axis)
        axes.plot([new_a], [self.equation(new_a)], 'ro')
        axes.plot([new_b], [self.equation(new_b)], 'ro')
        plt.show(block=False)




    def plot_x_versus_iterations(self):
        x = self.x
        k = self.k

        new_a = self.new_a
        new_b = self.new_b

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

        axes.plot(x_axis, y_axis)

        for i in range(0,len(y_axis)):
            plt.plot(x_axis[i], y_axis[i], 'ro')
        plt.show(block=False)




class bounding_phase_method(basic_optimization):
    def __init__(self, a, b, maximize, part):
        super().__init__(a, b, maximize, part)
        self.x = [random.uniform(a, b)]
        self.delta = random.uniform(10**-9, 10**-12)

    def minimize(self):
        out = open(f"./outputs/bounding_phase_method_part{self.part}.out", "w")
        k = 0
        a = self.a
        b = self.b
        x = self.x
        delta = self.delta
        
        f_x_minus_delta = super().equation(x[0]-delta)
        f_x = super().equation(x[0])
        f_x_plus_delta = super().equation(x[0]+delta)

        while True:
            if f_x_minus_delta >= f_x and f_x >= f_x_plus_delta :
                delta = abs(delta)
                break
            elif f_x_minus_delta <= f_x and f_x <= f_x_plus_delta :
                delta = -1 * abs(delta)
                break
            else :
                x = [random.uniform(a, b)]
                delta = random.uniform(10**-9, 10**-15)

            f_x_minus_delta = super().equation(x[0]-delta)
            f_x = super().equation(x[0])
            f_x_plus_delta = super().equation(x[0]+delta)

        out.write(f"Solving part with Bounding Phase Method- {self.part} \n a : {a} \n b : {b} \n delta : {delta} \n")

        out.write(f"#It\t\t\tx\t\t\tf_x\n")

        x.append(x[k] + ((2**k) * delta))

        f_x_k_plus_one = super().equation(x[k+1])
        f_x_k = super().equation(x[k])

        while(f_x_k_plus_one<=f_x_k):
            
            out.write(f"{k}\t\t{truncate_decimals(x[k])}\t\t{truncate_decimals(f_x_k)}\n")
            print(f"X value for k : {k} and x : {x[k]}")
            k=k+1
            x.append(x[k] + ((2**k) * delta))
            
            f_x_k = f_x_k_plus_one
            f_x_k_plus_one = super().equation(x[k+1])

        out.write(f"{k}\t\t{truncate_decimals(x[k])}\t\t{truncate_decimals(f_x_k)}\n")
        print(f"X value for k : {k} and x : {x[k]}")
        
        self.x = x
        self.delta = delta
        self.k = k
        self.new_a = self.x[self.k-1]
        self.new_b = self.x[self.k+1]

        out.write(f"Value for new a : {self.new_a} and new b : {self.new_b} after bounding phase method")

        out.close()


    def results(self):
        return self.x[self.k-1], self.x[self.k+1]



class interval_halving_method(basic_optimization):
    def __init__(self, a, b, maximize, part):
        super().__init__(a, b, maximize, part)
        self.epsilon = math.pow(10, -1*random.randint(3, 7))
        self.l = b-a
        self.x = []

    def minimize(self):
        out = open(f"./outputs/interval_halving_method_part{self.part}.out", "w")
        a = self.a
        b = self.b
        epsilon = self.epsilon
        l = self.l
        x_m = a + (b-a)/2
        self.x.append(x_m)
        k=0
        out.write(f"Continue Solving part - {self.part} with Interval Halving Method \n a : {a} \n b : {b} \n")
        out.write(f"#It\t\t\ta\t\t\tb\t\t\tx_m\n")

        while(abs(l)>epsilon):
            x_1 = a + l/4
            x_2 = b - l/4

            f_x_1 = super().equation(x_1)
            f_x_2 = super().equation(x_2)
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
            self.x.append(x_m)
            out.write(f"{k}\t\t\t{truncate_decimals(a)}\t\t\t{truncate_decimals(b)}\t\t\t{x_m}\n")
            print(f"A : {a}, B : {b} and X_M : {x_m}")
            k = k+1


        self.new_a = a
        self.new_b = b
        self.l = (b-a)
        self.k = k

        out.write(f"Value for new a : {self.new_a} and new b : {self.new_b} after interval halving method")

        out.close()

    def results(self):
        return self.new_a, self.new_b




def main():
    print("Enter a and b: ")
    a = float(input())
    b = float(input())

    print("Enter the part to be solved: ")
    part = int(input())

    if part>6:
        print("Please enter correct part to be solved!")
        return 0

    print("Wether the function has to be maximize or not. Enter Y/N: ")
    maxi = input()

    maxmize = False
    if maxi=="Y":
        maximize = True

    bounding_phase = bounding_phase_method(a, b, maximize, part)
    bounding_phase.minimize()
    a_bounding_phase, b_bounding_phase = bounding_phase.results()
    print(f"From bounding phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")
    bounding_phase.plot_range()
    bounding_phase.plot_x_versus_iterations()

    interval_halving = interval_halving_method(a_bounding_phase, b_bounding_phase, maximize, part)
    interval_halving.minimize()
    a_interval_halving, b_interval_halving = interval_halving.results()
    print(f"From interval halving phase method => a : {a_interval_halving}, b : {b_interval_halving}")
    interval_halving.plot_range()
    interval_halving.plot_x_versus_iterations()

    plt.show()

main()
