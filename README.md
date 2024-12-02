# Shopping Prediction Project

This project implements a k-nearest neighbors (k-NN) classifier to predict whether a user will make a purchase during an online shopping session based on various features of their browsing behavior.

## Features and Labels

- **Features (Evidence):**
  - `Administrative`: Number of administrative pages visited (integer).
  - `Administrative_Duration`: Time spent on administrative pages (float).
  - `Informational`: Number of informational pages visited (integer).
  - `Informational_Duration`: Time spent on informational pages (float).
  - `ProductRelated`: Number of product-related pages visited (integer).
  - `ProductRelated_Duration`: Time spent on product-related pages (float).
  - `BounceRates`: Bounce rate of the pages (float).
  - `ExitRates`: Exit rate of the pages (float).
  - `PageValues`: Value of the page (float).
  - `SpecialDay`: Closeness to a special day (float).
  - `Month`: Month of the session (converted to numeric, 0 for January to 11 for December).
  - `OperatingSystems`: Operating system used (integer).
  - `Browser`: Browser used (integer).
  - `Region`: User's region (integer).
  - `TrafficType`: Traffic source type (integer).
  - `VisitorType`: Returning visitor (1 for returning, 0 for new).
  - `Weekend`: Whether the session occurred on a weekend (1 for True, 0 for False).

- **Label:**
  - `Revenue`: Whether the user made a purchase (1 for Yes, 0 for No).

## Prerequisites

Ensure you have Python 3 installed on your system. You will also need to install the required Python package:

```bash
pip3 install scikit-learn
```

## Usage

1. Clone or download the project files.
2. Place the `shopping.csv` dataset in the same directory as `shopping.py`.
3. Run the script with the following command:

```bash
python shopping.py shopping.csv
```

### Expected Output

The script will load the data, train the k-NN model, and display the following metrics:

- Number of correct and incorrect predictions.
- Sensitivity (True Positive Rate).
- Specificity (True Negative Rate).

Example:

```
Correct: 1000
Incorrect: 200
True Positive Rate: 75.00%
True Negative Rate: 90.00%
```

## Files

- `shopping.py`: Main script to train and test the k-NN classifier.
- `shopping.csv`: Dataset containing online shopping session data.

## Notes

- Ensure that the `shopping.csv` file follows the format expected by the script (as described in the Features and Labels section).
- The script splits the data into training and testing sets, using 40% of the data for testing by default.
- Adjustments can be made to the code to experiment with different test sizes or classifier parameters.

## License

This project is for educational purposes and does not include any proprietary components. Feel free to modify and use it as needed.

