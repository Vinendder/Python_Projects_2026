
# Terminal start screen
print('''
========================== Welcome to the Number Guessing Game! ==========================''')

# The correct number
correct_number = 73

# All functions below are used to check the user's guess and provide feedback
def check_guess(x):
    if too_high(x):
        print("Your guess is too high!")
    elif too_low(x):
        print("Your guess is too low!")
    elif near_high(x):
        print("Your guess is close but a little high!")
    elif near_low(x):
        print("Your guess is close but a little low!")
    elif correct(x):
       print("You guessed the correct number!")
    else:
        print("Your guess is not valid!, Try using integers only")

def too_high(x):
    return 101 <= x 

def too_low(x):
    return 26 >= x 

def near_high(x):
    return 73 < x <= 100

def near_low(x):
    return 27 <= x < 73

def correct(x):
    return x == 73

# Life system for the game, wrong guess is punshied. Game ends when lives reach 0 or the user guesses the correct number
lives = 5
game_over = False

# While loop to keep the game running until the user guesses the correct number or runs out of lives
while game_over is False:
    user_guess_str = input("Guess a number! ")
    if user_guess_str.lstrip('-').isdigit():
        user_guess = int(user_guess_str)
        check_guess(user_guess)
        if user_guess == correct_number:
            print("You win!")
            game_over = True
        else:
            lives -= 1
            print(f"You have {lives} lives left!")
            if lives == 0:
                print("You lose!, Better luck next time!")
                game_over = True
    else:
        print("Your guess is not valid!, Try using integers only")
    