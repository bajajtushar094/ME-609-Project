# from single_variable_optimization import *
import numpy as np
import numdifftools as nd


class multi_optimization():
    def __init__(self, n, part, m):
        self.n = n
        self.x = np.array([0, 0])
        # print("X :", self.x)
        self.part = part
        self.m = m

    def equation(self, x):
        eqn = 0
        if self.part == 1:
            l = len(x)

            # eqn = np.matmul(np.transpose(x), x)

            for i in range(0, l-1):
                eqn = eqn + round(i*x[i]*x[i], 4)

        elif self.part == 2:
            eqn = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

        return eqn


class marquardt_method(multi_optimization):
    def __init__(self, n, part, m):
        self.ld = 10**6
        self.epsilon = 10**-3
        super().__init__(n, part, m)
        

    def minimize(self):
        k = 0
        x = self.x
        ld = self.ld
        print("Initial Value : ", x)

        
        while(True):
            print("Function Value : ", self.equation(x))
            f_grad = nd.Gradient(self.equation)(x)

            #print("Gradient : ", f_grad)

            f_norm = np.linalg.norm(f_grad)

            if f_norm<self.epsilon or k>=self.m:
                break
            
            Hessian_func = nd.Hessian(self.equation)

            hessian = Hessian_func(x)

            #print("Hessian : ", hessian)

            iden = np.dot(ld, np.identity(self.n, dtype=float))

            inverse = np.linalg.inv(np.add(hessian,iden))

            s_k = np.dot(-1, np.matmul(inverse, f_grad))

            x_plus_one = np.add(x, s_k)

            # while True:
            #     hessian = Hessian_func(x)

            #     #print("Hessian : ", hessian)

            #     iden = np.dot(ld, np.identity(self.n, dtype=float))

            #     inverse = np.linalg.inv(np.add(hessian,iden))

            #     s_k = np.dot(-1, np.matmul(inverse, f_grad))

            #     x_plus_one = np.add(x, s_k)

            #     if self.equation(x_plus_one)<self.equation(x):
            #         break

            #     self.ld = 2*self.ld
            #     #x = x_plus_one

            ld = ld/2
            x = x_plus_one
            k=k+1

        print("Iteration : ", k)

        self.x = x

    def results(self):
        return self.x, 



n = 2
part = 2
m = 3

mar = marquardt_method(n, part, m)
mar.minimize()
print(mar.results())










