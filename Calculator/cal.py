print("""
==================== Welcome To Calculator ====================""")

# Stores history
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
    if b == 0:
        return "Cannot modulo by 0"
    return a % b

def square_root(a):
    if a < 0:
        return "Root cannot be negative"
    return round(a ** 0.5, 5)


def square(a):
    return round(a**2, 5)

# List and dictionaries for operations 
basic_features = ["history", "clear", "exit"]

single_num_operations = {
    "root": square_root,
    "^2": square,
}

double_num_operations = {
    "+": add_numbers,
    "-": subtract_numbers,
    "*": multiply_numbers,
    "/": divide_numbers,
    "^": power_numbers,
    "modulo": modulus_numbers,
}

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
    if (operation not in basic_features
         and operation not in double_num_operations
         and operation not in single_num_operations): # Operation check
        print("Please enter valid operation.")
        continue
    elif operation == "exit":  # Basic features 
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
    elif operation in single_num_operations: # Single number operations
        result = single_num_operations[operation](first_number)
        print(f"The Result is {result}")
        history.append(f"{first_number} {operation} = {result}") 
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

    if operation in double_num_operations:
        result = double_num_operations[operation](first_number, second_number)
        print(f"The Result is {round(result, 10)}") # End result for operation of 2 numbers
        history.append(f"{first_number} {operation} {second_number} = {round(result, 10)}") # History for operation of 2 numbers 
    