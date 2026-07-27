import random


# Function for win/loss/tie count
def result_count(data):
    wins = 0
    losses = 0
    ties = 0

    for entry in data:
        if entry.endswith("Won"):
            wins += 1
        elif entry.endswith("Lost"):
            losses += 1
        elif entry.endswith("Tie"):
            ties += 1
    return wins, losses, ties


# Options for user and code
user_options = ["rock", "paper", "scissors", "history", "clear", "exit", "stats"]
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
        player_choice = input("""Enter Choice: Rock or Paper or Scissors or History or Clear or Stats or Exit 
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
        continue
    elif player_choice == "clear":
        if not match_history:
            print("History is already empty")
        else:
            match_history.clear()
            print("History cleared.")
        continue
    elif player_choice == "stats":
        wins, losses, ties = result_count(match_history)
        print(f"""
Wins: {wins}
Losses: {losses}
Ties: {ties}
 """)
        continue

    # All possible cases
    elif player_choice == computer_choice:
        print("Tie!")
        result = "Tie"
    elif player_wins[player_choice] == computer_choice:
        print("You win!")
        result = "Won"
    elif player_choice != computer_choice:
        print("You Lose :(")
        result = "Lost"

    match_history.append(f"{player_choice} vs {computer_choice} = {result}") # Adds history