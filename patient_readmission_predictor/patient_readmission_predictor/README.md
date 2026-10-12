# Patient Readmission Risk Predictor (Java)

Predicts the probability that a patient is readmitted to hospital within 30 days of discharge.
Pure Java 17+ with no external libraries: logistic regression is implemented from scratch.

## Run
```bash
javac *.java
java Main
```

Predict for a specific patient (age, length of stay, prior admissions, medications, diagnoses, diabetes, heart failure, emergency):
```bash
java Main 72 8 2 12 6 1 1 1
```

## How it works
1. `DataGenerator` creates 10,000 synthetic patient records with a realistic readmission pattern.
2. Data is split 80% train / 20% test.
3. `LogisticRegression` standardizes features and trains with gradient descent (L2 regularization).
4. `Metrics` reports accuracy, precision, recall, F1 and ROC AUC.
5. Each patient is labelled LOW (<15%), MEDIUM (15-35%) or HIGH (>35%) risk.

## Files
| File | Purpose |
|------|---------|
| `Patient.java` | Patient record and feature list |
| `DataGenerator.java` | Synthetic dataset generator |
| `LogisticRegression.java` | Model training and prediction |
| `Metrics.java` | Evaluation metrics |
| `Main.java` | Pipeline and command-line entry point |

## Ideas to extend
Load real data from CSV (e.g. the UCI Diabetes 130-US hospitals dataset), add more features,
try class weighting, build a Spring Boot REST API, or add a simple Swing UI.
