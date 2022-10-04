from single_variable_optimization import *
import numpy as np
import numdifftools as nd

class Multi_optimization():
    def __init__(self, n, part, m):
        self.n = n
        self.x = np.random.rand(n)
        # print("X :", self.x)
        self.part = part
        self.m = m

    def equation(self, x):
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


class Marquardt_method(Basic_optimization):
    def __init__(self, n, part, m):
        self.ld = 100
        self.epsilon = 10**-3
        self.x = np.random.rand(n)
        super().__init__(part, n, m)
        

    def minimize(self):
        k = 0
        x = self.x
        ld = self.ld
        
        while(True):
            f_grad = nd.Gradient(self.equation)(x)

            f_norm = np.linalg.norm(f_grad)

            if f_norm<=self.epsilon or k>=self.m:
                break
            
            Hessian_func = nd.Hessian(self.equation)

            hessian = Hessian_func(x)

            while True:
                hessian = Hessian_func(x)

                print(f"n = {self.n}")
                iden = np.dot(ld, np.identity(self.n, dtype=float))

                inverse = np.linalg.inv(np.add(hessian,iden))

                s_k = np.dot(-1, np.matmul(inverse, f_grad))

                # aplha = 1

                # alpha_s_k = np.dot(aplha, s_k)

                # eqn_aplha_s_k = self.equation(alpha_s_k)

                bounding_phase_method = Bounding_phase_method(self.part, self.n, self.m, x, s_k)
                bounding_phase_method.minimize()
                a_bounding_phase, b_bounding_phase = bounding_phase_method.results()

                print(f"--------------------------------------------------")
                print(f"Range from bounding phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")

                interval_halving_method = Interval_halving_method(a_bounding_phase, b_bounding_phase, self.part, self.n, self.m, x, s_k)
                interval_halving_method.minimize()
                a_interval_halving, b_interval_halving = interval_halving_method.results()
                
                alpha = a_interval_halving + (b_interval_halving-a_interval_halving)/2

                print(f"--------------------------------------------------")
                print(f"Range from interval halving phase method => a : {a_bounding_phase}, b : {b_bounding_phase}")

                x_plus_one = np.add(x, np.dot(alpha, s_k))

                if self.equation(x_plus_one)<=self.equation(x):
                    break

                ld = 2*ld
                #x = x_plus_one

            ld = ld/2
            x = x_plus_one
            k=k+1

        self.x = x

    def results(self):
        return self.x, 


def main():
    part = int(input("Enter a number between 1 and 6 to solve correspinding part of question: "))

    n = int(input("Enter n : "))

    m = int(input("Enter m : "))

    if part>6:
        print("Please enter correct part to be solved!")
        return 0

    print(f"--------------------------------------------------------------")

    marquardt = Marquardt_method(n, part, m)
    marquardt.minimize()

    print(f"--------------------------------------------------------------")
    print(f"Results from marquardt method : {marquardt.results()}")


main()
    











