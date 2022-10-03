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


class Marquardt_method(Multi_optimization):
    def __init__(self, n, part, m):
        self.ld = 100
        self.epsilon = 10**-3
        super().__init__(n, part, m)
        

    def minimize(self):
        k = 0
        x = self.x
        ld = self.ld
        print("Initial Value : ", x)

        
        while(True):
            print(f"X value at iteration {k} : ", x)
            print("Function Value : ", self.equation(x))
            f_grad = nd.Gradient(self.equation)(x)

            f_norm = np.linalg.norm(f_grad)

            print("F grad : ", f_grad)

            print("F norm : ", f_norm)

            if f_norm<=self.epsilon or k>=self.m:
                break
            
            Hessian_func = nd.Hessian(self.equation)

            hessian = Hessian_func(x)

            while True:
                hessian = Hessian_func(x)

                print("Hessian : ", hessian)

                iden = np.dot(ld, np.identity(self.n, dtype=float))

                print("Identity : ", iden)

                inverse = np.linalg.inv(np.add(hessian,iden))

                print("Inverse : ", inverse)

                s_k = np.dot(-1, np.matmul(inverse, f_grad))

                # aplha = 0.1

                # alpha_s_k = np.dot(aplha, s_k)

                # eqn_aplha_s_k = self.equation(alpha_s_k)

                print("S_k : ", s_k)

                print(f"X value before addition in iteration {k} : ", x)
                x_plus_one = np.add(x, s_k)
                print(f"X value after addition in iteration {k} : ", x)

                if self.equation(x_plus_one)<=self.equation(x):
                    break

                ld = 2*ld
                #x = x_plus_one

            ld = ld/2
            x = x_plus_one
            print("ld : ", ld)
            k=k+1

        print("Iteration : ", k)

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
    











