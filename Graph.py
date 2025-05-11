import matplotlib.pyplot as plt
import json

def readFromFile(filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        print(f"Contents of '{filename}':")
        return(data)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


models = ["mistralai--mistral-large-instruct", "anthropic--claude-3-haiku", "anthropic--claude-3-opus", "anthropic--claude-3.7-sonnet", "gpt-4o", "gpt-4o-mini", "o1", "o3-mini", "gemini-2.0-flash", "gemini-1.5-pro"]

path = input("Write filepath: ")
results = readFromFile(path)

plt.figure(figsize=(10, 6))

for model in models:
    model_results = [result for result in results if result[0] == model]    
    model_results = sorted(model_results,key=lambda x: x[1])

    model_tokens = [arr[1] for arr in model_results]
    model_times = [arr[2] for arr in model_results]
    
    plt.plot(model_tokens, model_times, label=model)

plt.xlabel('Anzahl der Output-Tokens')
plt.ylabel('Zeit (s)')
plt.title('Verarbeitungsdauer verschiedener LLMs in Abhängigkeit der generierten Output-Token')
plt.legend()
plt.tight_layout()
plt.show()