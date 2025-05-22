import pandas as pd
import numpy as np
data = pd.read_csv("find.csv")
def find_s_algorithm(data):
    """Implements the Find-S algorithm to find the most specific hypothesis."""
    # Extract feature columns and target column
    attributes = data.iloc[:, :-1].values  # All columns except last
    target = data.iloc[:, -1].values  # Last column (class labels)
    # Step 1: Initialize hypothesis with first positive example
    for i in range(len(target)):
        if target[i] == "Yes":  # Consider only positive examples
            hypothesis = attributes[i].copy()
            break
    # Step 2: Update hypothesis based on other positive examples
    for i in range(len(target)):
        if target[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attributes[i][j]:
                    hypothesis[j] = '?'  # Generalize inconsistent attributes
    return hypothesis
# Run Find-S Algorithm
final_hypothesis = find_s_algorithm(data)
# Print the learned hypothesis
print("Most Specific Hypothesis:", final_hypothesis)
