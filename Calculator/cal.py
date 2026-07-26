print("""
==================== Welcome To Calculator ====================""")

# Variables to compare and store
valid_operations = ["+", "-", "*", "/", "^", "modulo", "root", "^2", "exit", "history", "clear"]
history = []

# Main functions defined
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
    return round(a ** 0.5, 10)     


def square(a):
    return a**2

# Main loop for function
while True:
    while True:  # Secondary loop for first value
        try:  # Trying for numbers only
            first_number = float(input("Enter first number: "))
            break
        except ValueError:
            print("Please enter a valid number.")

    operation = input(""" Enter operation: +, -, *, /, ^, modulo, root, ^2, exit, history, clear
            >  """).strip().lower()
    if operation not in valid_operations: # Operation check
        print("Please enter valid operation.")
        continue
    elif operation == "exit":  # Single/No number operations
        break
    elif operation == "history":
        if not history:
            print("There is no history.")
        else:
            for i, entry in enumerate(history, start= 1):
                print(f"{i}) {entry}")
        continue
    elif operation == "clear":
        if not history:
            print("History already empty.")
        else:
            history.clear()
            print("History cleared.")
        continue
    elif operation == "root":
        if first_number < 0:
            print("Cannot find the square root of negative numbers")
        else:
            print(f"The Result is {square_root(first_number)}")
            history.append(f"{first_number} {operation} = {square_root(first_number)} ")
        continue
    elif operation == "^2":
        print(f"The Result is {round(square(first_number))}")
        history.append(f"{first_number} {operation} ={round(square(first_number))} ")
        continue

    while True: # Loop for second value to avoid going all the way back for false input
        try:
            second_number = float(input("Enter second number: "))
            if (operation == "/" or operation == "modulo") and second_number == 0:
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

    print(f"The Result is {round(result, 10)}") # End result for operation of 2 numbers
    history.append(f"{first_number} {operation} {second_number} = {round(result, 10)}") # History for operation of 2 numbers 
    