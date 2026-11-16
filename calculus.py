"""
Calculus Solver — CS50P-style project
--------------------------------------
Takes CLEAN symbolic math input from the user (validated, not raw eval())
and solves:
  1. Differentiation
  2. Integration (indefinite & definite)
  3. AOD — Application of Derivatives (critical points, max/min)
  4. Differential Equations (basic ODEs)

Built on SymPy, Python's symbolic mathematics library — it manipulates
expressions exactly (2*x, not 2.000001*x), which is what calculus needs.
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

x = sp.symbols('x')

# Lets users type "2x" instead of "2*x", and "x^2" instead of "x**2"
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def clean_parse(expr_str, local_dict=None):
    """
    Safely turn a string into a SymPy expression.
    Raises ValueError with a friendly message on bad input,
    instead of letting a raw exception (or worse, eval()) through.
    """
    if not expr_str.strip():
        raise ValueError("Input was empty.")
    try:
        expr = parse_expr(expr_str, transformations=TRANSFORMATIONS, local_dict=local_dict)
        return expr
    except Exception as e:
        raise ValueError(f"Couldn't understand '{expr_str}': {e}")


def differentiate():
    expr = clean_parse(input("Enter f(x): ").strip())
    order_str = input("Order of derivative [default 1]: ").strip()
    order = int(order_str) if order_str else 1
    if order < 1:
        raise ValueError("Order must be a positive integer.")
    result = sp.simplify(sp.diff(expr, x, order))
    print(f"\nd^{order}/dx^{order} [{expr}] = {result}\n")


def integrate():
    expr = clean_parse(input("Enter f(x): ").strip())
    definite = input("Definite integral? (y/n): ").strip().lower() == 'y'
    if definite:
        a = clean_parse(input("Lower limit: ").strip())
        b = clean_parse(input("Upper limit: ").strip())
        result = sp.integrate(expr, (x, a, b))
        print(f"\n∫ from {a} to {b} of ({expr}) dx = {result}\n")
    else:
        result = sp.integrate(expr, x)
        print(f"\n∫ ({expr}) dx = {result} + C\n")


def aod_analysis():
    """Application of Derivatives: critical points + max/min classification."""
    expr = clean_parse(input("Enter f(x): ").strip())
    dydx = sp.diff(expr, x)
    d2ydx2 = sp.diff(expr, x, 2)
    critical_points = sp.solve(sp.Eq(dydx, 0), x)

    print(f"\nf'(x)  = {dydx}")
    print(f"f''(x) = {d2ydx2}")

    if not critical_points:
        print("No critical points found.\n")
        return

    print("Critical points:")
    for cp in critical_points:
        val = d2ydx2.subs(x, cp)
        val = sp.nsimplify(val)
        if val.is_real is False:
            nature = "complex — skipped"
        elif val > 0:
            nature = "local minimum"
        elif val < 0:
            nature = "local maximum"
        else:
            nature = "inconclusive (2nd derivative test fails, check manually)"
        print(f"  x = {cp}  ->  {nature}")
    print()


def solve_ode():
    print("\nUse y(x) for the function and Derivative(y(x), x) for dy/dx.")
    print("Example: Eq(Derivative(y(x), x), y(x) + x)\n")
    y_func = sp.Function('y')
    local_dict = {'y': y_func, 'x': x, 'Eq': sp.Eq, 'Derivative': sp.Derivative}
    ode = clean_parse(input("ODE: ").strip(), local_dict=local_dict)
    solution = sp.dsolve(ode, y_func(x))
    print(f"\nSolution: {solution}\n")


def main():
    menu = {
        '1': ("Differentiation", differentiate),
        '2': ("Integration", integrate),
        '3': ("AOD — critical points / max-min", aod_analysis),
        '4': ("Differential Equations", solve_ode),
    }
    while True:
        print("=== Calculus Solver ===")
        for key, (label, _) in menu.items():
            print(f"{key}. {label}")
        print("0. Exit")

        choice = input("Choose an option: ").strip()
        if choice == '0':
            print("Goodbye.")
            break
        elif choice in menu:
            try:
                menu[choice][1]()
            except ValueError as e:
                print(f"Input error: {e}\n")
            except Exception as e:
                print(f"Something went wrong: {e}\n")
        else:
            print("Invalid choice — pick a number from the menu.\n")


if __name__ == "__main__":
    main()