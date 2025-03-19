[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/2ciRUaGA)
# Cross-Validation Implementation Exercise

## Overview
In this exercise, you will implement a cross-validation function that supports both k-fold and Leave-One-Out (LOO) cross-validation methods. The function evaluates machine learning models by splitting data into training and validation sets.

## Task Description
Complete the `cross_validation` function in `Lab2_6_CV.py` by implementing the following TODOs:

1. Calculate `fold_size` based on the number of folds
2. Initialize a list to store accuracy scores
3. For each fold:
    - Generate validation set indices
    - Generate training set indices
    - Split data into training and validation sets
    - Train the model
    - Calculate and store accuracy scores
4. Return mean and standard deviation of accuracy scores

## Testing
Your implementation will be tested against known results using:
- 5-fold cross-validation (expected mean ≈ 0.95, std ≈ 0.044)
- LOO cross-validation (expected mean ≈ 0.93, std ≈ 0.255)

## Tips
- Use numpy arrays for data manipulation
- Remember that for LOO CV, `nFolds` equals the number of samples
- Ensure proper data splitting to avoid overlap between training and validation sets