# Heart Disease Classification

This project is a Python implementation of various machine learning classifiers for predicting the presence of heart disease in patients based on their clinical data. The classifiers used in this project include Support Vector Machine (SVM), Naive Bayes, Decision Tree, and K-Nearest Neighbors (KNN).

## Dataset

The project uses the "Heart Disease Classification Dataset" from Kaggle. The dataset contains various features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, and more. The target variable indicates the presence or absence of heart disease. CSV file of dataset is given in the repository.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yashshah035/Classifier_Models.git

2. Install the required dependencies:

   ```bash
   pip install numpy pandas scikit-learn

## Usage

1. Navigate to the project directory:

   ```bash
   cd Classifier_Models

2. Run the main Python file:

   ```bash
   python demo_ml_classifier.py

3. The script will load the dataset, preprocess the data, and present you with a menu to choose the classifier you want to use.

4. Select the classifier by entering the corresponding number:
1. Support Vector Machine
2. Naive Bayes
3. Decision Tree
4. K-Nearest Neighbors

5. The script will train the chosen classifier on the training data and evaluate its performance on the test data, displaying the accuracy score.

6. You can repeat the process by selecting a different classifier or exit the program by entering 'q'.

## Code Explanation

The provided code performs the following steps:

1. Imports the necessary libraries: Pandas, Scikit-learn classifiers, and other required modules.
2. Loads the dataset from the CSV file.
3. Performs data preprocessing steps, such as handling missing values and encoding categorical features.
4. Splits the data into training and testing sets.
5. Defines a `Classifier` class with methods for each classifier (SVM, Naive Bayes, Decision Tree, and KNN).
6. Implements a loop that prompts the user to select a classifier.
7. Trains the chosen classifier on the training data and evaluates its performance on the test data, printing the accuracy score.
8. Allows the user to select a different classifier or exit the program.
## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
