def add_numbers(operation):
    return first_number + second_number


def subtract_numbers(operation):
    return first_number - second_number


def multiply_numbers(operation):
    return first_number * second_number


def divide_numbers(operation):
    return first_number / second_number


def power_numbers(operation):
    return first_number**second_number


def modulus_numbers(operation):
    return first_number % second_number


def square_root(operation):
    return first_number**0.5


def square(operation):
    return first_number**2


Calculator_off = False

while not Calculator_off:
    first_number = float(input("Enter first number "))
    if not first_number.is_integer():
        print("Please enter a valid number.")
        continue

    valid_operations = ["+", "-", "*", "/", "^", "modulo", "root", "^2", "exit"]
    operation = input("""Enter operation (+, -, *, /, ^, modulo, root, ^2, exit):
        """).lower()
    if operation not in valid_operations:
        print("Please enter a valid operation.")
        continue
    elif "root" in operation:
        result = square_root(operation)
        print(f"The result is: {result}")
        continue
    elif "^2" in operation:
        result = square(operation)
        print(f"The result is {result}")
        continue
    elif "exit" in operation:
        break

    second_number = float(input("Enter second number "))
    if not second_number.is_integer():
        print("Please enter a valid number.")
        continue

    if "+" in operation:
        result = add_numbers(operation)
        print(f"The result is {result}")
    elif "-" in operation:
        result = subtract_numbers(operation)
        print(f"The result is {result}")
    elif "*" in operation:
        result = multiply_numbers(operation)
        print(f"The result is {result}")
    elif "/" in operation:
        result = divide_numbers(operation)
        print(f"The result is {result}")
    elif "^" in operation:
        result = power_numbers(operation)
        print(f"The result is {result}")
    elif "modulo" in operation:
        result = modulus_numbers(operation)
        print(f"The result is {result}")
