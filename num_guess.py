
# Terminal start screen

print('''
========================== Welcome to the Number Guessing Game! ==========================''')

# Values for correct number, lives and level
correct_number = 73
lives = 5
level = 1

# Main function body
def guess_check(x):
    if correct_guess(x):
        print("Congratulations! You've guessed the correct number!")
    elif very_close(x):
        print("You're very close! Try again.")
    elif close(x):
        print("You're close! Keep trying.")
    elif far(x):
        print("You're far off! Keep guessing.")
    else:
        print("You're way off! Try again.")

# Each function defined
def correct_guess(x):
    return x == correct_number

def very_close(x):
    return abs(x - correct_number) <= 5

def close(x):
    return abs(x - correct_number) <= 10

def far(x):
    return abs(x - correct_number) <= 25

# Main game loop 
game_over = False
while not game_over and level <= 5:                 # =========== Leveling system =============
    user_guess_str = input("Guess a number ")       # Ps: This is an input checker
    if not user_guess_str.lstrip('-').isdigit():
        print("Please enter a valid number.")
        continue
    user_guess = int(user_guess_str)
    guess_check(user_guess)

    if correct_guess(user_guess):
        print(f"{level} level completed! You have {lives} lives remaining.")
        level += 1
        correct_number = correct_number + (level * 5)
    else:
        lives -= 1
        print(f"Wrong guess! You have {lives} lives remaining.")
        if lives == 0:
            game_over = True
            print("Game Over! But you can try again ;)")

if level > 5 and lives <= 4:
    print("Congratulations! You have guessed all numbers correctly")
elif level > 5 and lives == 5:                          # Easter egg :>
    print("Yo bro stop hacking")

    


