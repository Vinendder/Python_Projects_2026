import random

# Options for user and code
user_options = ["rock", "paper", "scissors", "history", "clear", "exit"]
computer_options = ["rock", "paper", "scissors"]
match_history = []

# Comparison user vs code
player_wins = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}

# Main loop
while True:
    computer_choice = random.choice(computer_options)

    while True: # loop for player choice
        player_choice = input("""Enter Choice: Rock or Paper or Scissors or History or Clear or Exit
                                 > """).strip().lower()
        if player_choice not in user_options:
            print("Enter Valid Choice")
            continue
        else:
            break

    # Basic features
    if player_choice == "exit":
        break
    elif player_choice == "history":
        if not match_history:
            print("History is empty.")
        else:
            for i, entry in enumerate(match_history, start= 1):
                print(f"{i}) {entry}")
    elif player_choice == "clear":
        if not match_history:
            print("History is already empty")
        else:
            match_history.clear()
            print("History cleared.")

    # All possible cases
    elif player_choice == computer_choice:
        print("Tie!")
        match_history.append(f"{player_choice} vs {computer_choice} = Draw")
    elif player_wins[player_choice] == computer_choice:
        print("You win!")
        match_history.append(f"{player_choice} vs {computer_choice} = Won")
    elif player_choice != computer_choice:
        print("You Lose :(")
        match_history.append(f"{player_choice} vs {computer_choice} = Lost")

