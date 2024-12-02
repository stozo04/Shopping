import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    # Dictionary to map month abbreviations to numeric values (0 for January, 11 for December)
    month_map = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    # List to store features for each user session as numeric values
    evidence = []
    # List to store binary labels indicating purchase intent (1 for true, 0 for false)
    labels = []

    with open(filename, mode="r") as file:
        # Use DictReader to read the CSV into a dictionary for easy access to column names
        reader = csv.DictReader(file)
        for row in reader:
            # Process each row, convert to the required numeric format, and append to evidence
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_map[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            # Extract the target label (Revenue) and convert to binary
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Initialize k-nearest neighbors classifier with k=1 and fit the model
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).
    """
    # Calculate the true positives and true negatives
    true_positive = sum(1 for actual, predicted in zip(
        labels, predictions) if actual == 1 and predicted == 1)
    true_negative = sum(1 for actual, predicted in zip(
        labels, predictions) if actual == 0 and predicted == 0)

    # Calculate the total positives and negatives in the actual labels
    total_positive = sum(1 for actual in labels if actual == 1)
    total_negative = sum(1 for actual in labels if actual == 0)

    # Sensitivity: Proportion of actual positives correctly identified
    sensitivity = true_positive / total_positive if total_positive > 0 else 0

    # Specificity: Proportion of actual negatives correctly identified
    specificity = true_negative / total_negative if total_negative > 0 else 0

    return sensitivity, specificity


if __name__ == "__main__":
    main()
