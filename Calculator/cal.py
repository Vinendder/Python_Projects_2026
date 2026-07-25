print("""
==================== Welcome To Calculator ====================""")


def add_numbers(a, b):
    return a + b


def subtract_numbers(a, b):
    return a - b


def multiply_numbers(a, b):
    return a * b


def divide_numbers(a, b):
    return a / b


def power_numbers(a, b):
    return a**b


def modulus_numbers(a, b):
    return a % b


def square_root(a):
    return a**0.5


def square(a):
    return a**2

while True:
    while True:
        try:
            first_number = float(input("Enter first number: "))
            break
        except ValueError:
            print("Please enter a valid number.")

    valid_operations = ["+", "-", "*", "/", "^", "modulo", "root", "^2", "exit"]
    operation = input(""" Enter operation: +, -, *, /, ^, modulo, root, ^2, exit
            >  """).lower()
    if operation not in valid_operations:
        print("Please enter valid operation.")
        continue
    elif operation == "exit":
        break
    elif operation == "root":
        print(f"The Result is {square_root(first_number)}")
        continue
    elif operation == "^2":
        print(f"The Result is {square(first_number)}")
        continue

    while True:
        try:
            second_number = float(input("Enter second number: "))
            if operation == "/" and second_number == 0:
                print("Cannot divide by 0")
                continue
            break
        except ValueError:
            print("Please enter a valid number")

    if operation == "+":
        result = add_numbers(first_number, second_number)
    elif operation == "-":
        result = subtract_numbers(first_number, second_number)
    elif operation == "*":
        result = multiply_numbers(first_number, second_number)
    elif operation == "/":
        result = divide_numbers(first_number, second_number)
    elif operation == "^":
        result = power_numbers(first_number, second_number)
    elif operation == "modulo":
        result = modulus_numbers(first_number, second_number)

    print(f"The Result is {result}")
    
    