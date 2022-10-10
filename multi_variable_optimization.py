from single_variable_optimization import *
import numpy as np
import numdifftools as nd
import pandas as pd
import time
import matplotlib.pyplot as plt

class Multi_optimization():
    def __init__(self, part, n, m, x):
        self.n = n
        self.x = np.random.rand(n)
        print("Initial value for x : ", self.x)
        #self.x = np.array(x)
        self.part = part
        self.m = m

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

                



class Marquardt_method(Multi_optimization):
    def __init__(self, part, n, m, x):
        self.ld = 100
        self.epsilon = 10**-3
        #self.x = np.random.rand(n)
        super().__init__(part, n, m, x)
        

    def minimize(self):
        k = 0
        n = self.n
        x = self.x
        ld = self.ld
        func_eva = 0
        
        while(True):
            f_grad = self.gradient(x)
            func_eva += 2*n

            f_norm = np.linalg.norm(f_grad)

            if f_norm<=self.epsilon or k>=self.m:
                break
            
            f_x = self.equation(x)
            func_eva+=1

            while True:
                hessian_mat = self.hessian(x)
                func_eva += 2*n*n + 1

                print(f"n = {self.n}")
                iden = np.dot(ld, np.identity(self.n, dtype=float))

                inverse = np.linalg.inv(np.add(hessian_mat,iden))

                s_k = np.dot(-1, np.matmul(inverse, f_grad))

                # bounding_phase_method = Bounding_phase_method(self.part, x, s_k)
                # bounding_phase_method.minimize()
                # a_bounding_phase, b_bounding_phase, func_eva_bounding_phase = bounding_phase_method.results()

                # func_eva+=func_eva_bounding_phase
                # print(f"--------------------------------------------------")
                # print(f"Range from bounding phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")

                # interval_halving_method = Interval_halving_method(self.part, x, s_k, a=a_bounding_phase, b=b_bounding_phase)
                # interval_halving_method.minimize()
                # a_interval_halving, b_interval_halving, func_eva_interval_halving = interval_halving_method.results()
                
                # func_eva+= func_eva_interval_halving
                
                # alpha = a_interval_halving + (b_interval_halving-a_interval_halving)/2

                # print(f"--------------------------------------------------")
                # print(f"Range from interval halving phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")

                #x_plus_one = np.add(x, np.dot(alpha, s_k))

                x_plus_one = np.add(x, s_k)

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

    def results(self):
        return self.x, self.k, self.func_eva



def histogram(x_axis, y_axis, part, ylabel):
    #plt.ylim(min(y_axis)-1, max(y_axis)+2)
    plt.xlabel("Dimension of input")
    plt.ylabel("Number of {ylabel}")
    plt.title(f"Plot of dimesion vs iterations for question {part}")
    plt.bar(x_axis, y_axis, color='blue', width=0.3)
    plt.savefig(f"./phase_2_graphs/bar_plots/{ylabel}/question_{part}.png")


def main():
    df = pd.read_csv('./ME609_Project_rough.csv')
    xs = []
    itrs = []
    func_evas = []
    
    for i, row in df.iterrows():
        part, n, m = row['part'], row['n'], row['m']

        if part>5:
            print("Value of part should be less than or equals to 5")
            continue

        print(f"--------------------------------------------------------------")

        marquardt = Marquardt_method(part, n, m, [1, -1, 1, -1, 1])
        marquardt.minimize()

        print(f"--------------------------------------------------------------")
        x, itr, func_eva = marquardt.results()
        xs.append(x)
        itrs.append(itr)
        func_evas.append(func_eva)
        print(f"Results from marquardt method for row {i+1}: {x}")
        print(f"Iterations for part {part} : {itr}")
        print(f"function evaluation for part {part} : {func_eva}")

        time.sleep(2)

def create_histogram_plots():

    for i in range(1, 6):
        itrs=[]
        func_evas = []
        for j in range(1, 6):
            print(f"--------------------------------------------------------------")
            marquardt = Marquardt_method(i, j, 100)
            marquardt.minimize()
            print(f"--------------------------------------------------------------")
            x, itr, func_eva = marquardt.results()
            itrs.append(itr)
            func_evas.append(func_eva)
            print(f"Results from marquardt method for row {i+1}: {x}")

        histogram([1,2,3,4,5], itrs, i, "iterations")
        histogram([1,2,3,4,5], func_evas, i, "function evaluations")

if __name__ == "__main__":
    main()
    #create_histogram_plots()










