import pandas as pd
import json
import time
import os

def readFromFile(filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        print(f"Contents of '{filename}':")
        return(data)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

def writeToFile():
    filename = "descriptive_stats.txt"
    try:
        with open(filename, "x") as file:
            file.write("Descriptive Statistics by Model:\n\n")
            file.write(stats)
    except FileExistsError:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        new_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
        with open(new_filename, "x") as file:
            file.write("Descriptive Statistics by Model:\n\n")
            file.write(stats)
        print(f"The file '{new_filename}' was created successfully.")
    else:
        print(f"The file '{filename}' was created successfully.")

path = input("Write filepath: ")
data = readFromFile(path)

dataframe = pd.DataFrame(data, columns=["model", "output_tokens", "time_seconds", "tokens_per_time"])

grouped = dataframe.groupby("model")

stats = grouped.describe().to_string()

writeToFile()