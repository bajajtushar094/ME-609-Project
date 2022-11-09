from multi_variable_optimization import *

class Constraint_optimization():
    def __init__(self, part, n, user_input, x):
        self.part = part
        self.n = n
        self.user_input = user_input
        self.x = np.array(x)
        self.c = 10
        self.r = 0.1

    def bracket_operator(self, x):
        if x<0:
            return x
        else:
            return 0


    def equation(self, x, r):
        part = self.part

        # eqn = 0
        # if part==1:
        #     eqn = (((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)+(r*(self.bracket_operator(((x[0]-5)**2)+(x[1]**2)-26))**2))
        # elif part==2:
        #     eqn = (x[0] - 10)**3+(x[1] - 20)**3 + r*(self.bracket_operator((x[0]-5)**2 + (x[1]-5)**2 -100)) + r*(self.bracket_operator(-1*((x[0] - 6)**2 + (x[1] - 5)**2 - 82.81)))

        # return eqn

        nc = 1 ## number of constraints. 
        g = np.zeros(nc) 

        sum_ = pow((pow(x[0],2) + x[1] - 11),2) + pow((pow(x[1],2) + x[0] - 7),2)
        g[0] = -26.0 + pow((x[0]-5.0), 2) + pow(x[1],2);#constraints.

        for i in range(nc):
            if(g[i] < 0.0): ## meaning that the constraint is violatd.
                sum_ = sum_ + r*g[i]*g[i]

        return sum_


class Penalty_function_method(Constraint_optimization):
    def __init__(self, part, n, user_input, x):
        print(f"received x value : {type(x)}")
        super().__init__(part, n, user_input, x)

    def minimize(self):
        x = self.x
        part = self.part
        n = self.n
        r = self.r
        c = self.c
        r_array = [r]   

        k=1

        marquardt_method = Marquardt_method(part, n, 100, 'N', x, r)
        marquardt_method.minimize()

        x_plus_one, itrs, func_eval = marquardt_method.results()
        x = x_plus_one
        print(f"x_1 : {x_plus_one}")
        r_array.append(c*r)

        while True:
            marquardt_method = Marquardt_method(part, n, 100, 'N', x, r_array[-1])
            marquardt_method.minimize()

            x_plus_one, itrs, func_eval = marquardt_method.results()

            p = self.equation(x, r_array[-2])
            print(f"Penalty function value at x_0 : {p}")
            p_plus_one = self.equation(x_plus_one, r_array[-1])
            print(f"Penalty function value at x_1 : {p_plus_one}")

            if abs(p_plus_one-p)<(10**-3):
                break

            r_array.append(c*r_array[-1])
            x = x_plus_one
            k+=1
            break

        self.x = x_plus_one
    

    def results(self):
        return self.x


def main():
    penalty_function_method = Penalty_function_method(1, 2, 'N', [0,0])

    penalty_function_method.minimize()
    x = penalty_function_method.results()

    print(f"final answer from penalty method -> x : {x}")

if __name__ == "__main__":
    main()
            