import random
import math
from re import A, X
import numpy as np
import matplotlib.pyplot as plt

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


class bounding_phase_method(basic_optimization):
    def __init__(self, a, b, maximize, part):
        super().__init__(a, b, maximize, part)
        # self.x0 = random.uniform(a, b)
        self.x = [random.uniform(a, b)]
        self.delta = random.uniform(10**-9, 10**-12)
        # self.x_array = [0.60]
        # self.delta = [0.50]
        # print(f"delta : {self.delta}")

    def minimize(self):
        k = 0
        a = self.a
        b = self.b
        x = self.x
        delta = 10**-19

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
                delta = random.uniform(10**-9, 10**-12)

            f_x_minus_delta = super().equation(x[0]-delta)
            f_x = super().equation(x[0])
            f_x_plus_delta = super().equation(x[0]+delta)

        # if f_x_minus_delta >= f_x and f_x >= f_x_plus_delta :
        #     delta = abs(delta)
        # elif f_x_minus_delta <= f_x and f_x <= f_x_plus_delta :
        #     delta = -1 * abs(delta)

        x.append(x[k] + ((2**k) * delta))

        while(super().equation(x[k+1])<=super().equation(x[k])):
            

            print(f"X value for k : {k} and x : {x[k] + ((2**k) * delta)}")

            x.append(x[k] + ((2**k) * delta))
            k = k+1

        self.x = x
        self.delta = delta
        self.k = k


    def results(self):
        return self.x[self.k-1], self.x[self.k+1]



class interval_halving_method(basic_optimization):
    def __init__(self, a, b, maximize, part):
        super().__init__(a, b, maximize, part)
        self.epsilon = math.pow(10, -1*random.randint(3, 7))
        self.l = b-a
        self.x_array = []

    def minimize(self):
        a = self.a
        b = self.b
        epsilon = self.epsilon
        l = self.l
        x_m = a + (b-a)/2

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

            print(f"A : {a}, B : {b} and X_M : {x_m}")


        self.a = a
        self.b = b
        self.l = (b-a)

    def results(self):
        return self.a, self.b


test1 = bounding_phase_method(-10, 0, True, 1)
# test1 = interval_halving_method(-2, 1, True, 2)

test1.minimize()

x_k_minus_one, x_k_plus_one = test1.results()

print(f"a : {x_k_minus_one}, b : {x_k_plus_one}")

a = []
b = []
for i in range(-10,0):
    y = test1.equation(i)
    a.append(i)
    b.append(y)

fig= plt.figure()
axes=fig.add_subplot(111)
axes.plot(a,b)
axes.plot([x_k_minus_one], [test1.equation(x_k_minus_one)], 'bo')
axes.plot([x_k_plus_one], [test1.equation(x_k_plus_one)], 'bo')
plt.show()




