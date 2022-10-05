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