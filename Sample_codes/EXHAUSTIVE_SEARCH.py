
def Exhaustive_Search(a,b):
    print("**********************************")
    print("Exhaustive Search Method")
    n = float(input("Enter 'n' = number of steps "))

    # Step 1
    delta = (b - a) / n                                 # Set delta value
    x1 = a                                              # New points
    x2 = x1 + delta
    x3 = x2 + delta

    feval = 0                                           #  function evaluation
    f1 = objective_function(x1)                         #  Calculate objective_function
    f2 = objective_function(x2)
    f3 = objective_function(x3)
    i = 1                                               #  Number of iterations
    feval = feval + 3

    if x3 > b:
        print(f"No minimum exist in the interval ( {a,b} ) for the given objective function")
        return

    out = open(r"Exhaustive_Search_iterations.out", "w")  # Output file
    #out = open(r"Exhaustive_Search_iterations.txt", "w")  # Output file

    out.write("#It\t\tx1\t\tx2\t\tx3\t\tf(x1)\t\tf(x2)\t\tf(x3)")
    out.write("\n")

    while x3 <= b:
        out.write( str(i)+"\t\t"+ str(round(x1,4))+"\t\t"+ str(round(x2,4)) +"\t\t"+str(round(x3,4)) +"\t\t"+str(round(f1,4)) +"\t\t"+str(round(f2,4)) +"\t\t"+str(round(f3,4)) )
        out.write("\n")

        if f1 >= f2:                                    # Termination condition
            if f2 <= f3:
                break

                                                        # If not terminated, update values
        x1 = x2
        x2 = x3
        x3 = x2 + delta
        f1 = f2
        f2 = f3
        f3 = objective_function(x3)
        feval = feval+1
        i = i + 1
        # Step 3    and end of while loop

    print("**********************************")
    print(f"The minimum point lies between ( {x1,x3} )")
    print("Total number of function evaluations: " + str(feval));

    out.write("\n")
    out.write("The minimum point lies between  "+ str(round(x1,4)) +" and " + str(round(x3,4)))    # Store in the file
    out.write("\n")
    out.write("Total number of function evaluations: " + str(round(feval,4)))


    out.close()



def objective_function(x):
	return(x*x + (54/x))




print("Enter ")                                          # Ranges of x
a = float(input("a = lower limit of x "))
b = float(input("b = upper limit of x "))

Exhaustive_Search(a,b)

