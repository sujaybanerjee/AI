"""
CS311 Programming Assignment 5: Decision Trees

Full Name: Sujay Banerjee

Description of performance on Hepatitis data set:

The decision tree model performs well on predicting hepatitis survival, with an accuracy of 86.5%, 
recall of 90.5%, precision of 92.7%, and an F1 score of 91.6%, meaning it's generally accurate and reliable. 
Key splits occur on features like varices (a serious liver disease complication) and histology (microscopic liver exam results), 
both expected indicators of disease severity and survival outcomes. It also uses SGOT (enzyme) and bilirubin (waste product) 
levels—-high values here can signal liver damage. Overall, the model's structure makes sense and give us a way to look at how 
different clinical factors influence patient outcomes.

Description of Adult dataset discretization and selected features:

Age was grouped into ranges like "0-19," "20-39," "40-59," and so on, up to "100+." 
Capital-gain and capital-loss were split into progressively higher ranges, with capital-gain spanning "0-10000" up to "100000+," 
and capital-loss grouped from "0-1000" up to "5000+." Selected features included education, hours-per-week, sex, 
occupation, marital-status, and age.

Potential effects of using your model for marketing a cash-back credit card:

Using this model to target high-earners for credit card marketing raises some concerns about fairness and relevance. 
Since it relies on 1994 demographic data, it may reflect income patterns and societal biases that are slightly outdated,
potentially leading to exclusion of certain groups. Also, using features like marital status, sex, or occupation could result in a 
skewed targeting approach, overlooking high-earners who aren't the typical profile.  
High-earners are likely to appreciate rewards tailored to their spending habits, enhancing customer loyalty and satisfaction.
This offer might primarily benefit those who already have higher incomes, potentially leaving out low- to moderate-income customers
 who could also benefit from cash-back incentives. This could lead to negative perceptions of the brand and its marketing practices.

"""

import argparse, os, random, sys
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]


class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted labeled for array-like example x"""
        # Implement prediction based on value of self.attr in x

        # get the value of the attribute the branch 
        attr_value = x[self.attr]
        
        # get the subtree for this value
        subtree = self.branches.get(attr_value)

        # Recursively predict using the appropriate subtree
        return subtree.predict(x)

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)


class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)


def entropy(probs):
    """Calculate entropy given a list of probabilities"""
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr"""
    # Return information gain metric for selecting attributes

    # calculate entropy for the whole dataset before splitting
    base_entropy = entropy(y.value_counts(normalize=True))  # B(p / (p + n)) 
    # print(f"Base entropy: {base_entropy}")

    # calculate weighted entropy for each subset of attribute attr
    remainder = 0.0
    for value in X[attr].unique():  # unique values of the attribute and their counts
        subset = y[X[attr] == value]
        subset_prob = len(subset) / len(X)
        subset_entropy = entropy(subset.value_counts(normalize=True))
        # print(f"Subset entropy for {attr} = {value}: {subset_entropy}") 
        remainder += subset_prob * subset_entropy

    # return the information gain (reduction in entropy)
    return base_entropy - remainder



def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.Series,
) -> DecisionNode:
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """
    # Implement recursive tree construction based on pseudo code in class and the assignment

    # Base Case 1: If examples is empty, return plurality label (.mode()) of parent examples
    if X.empty:
        return DecisionLeaf(y_parent.mode()[0]) 

    # Base Case 2: If all examples have the same label, return that label
    if len(y.unique()) == 1:
        return DecisionLeaf(y.iloc[0]) 

    # Base Case 3: If no attributes left, return plurality label of examples
    if not attrs:
        return DecisionLeaf(y.mode()[0])

    # Recursive Case: choose attribute with max information gain
    gains = {attr: information_gain(X, y, attr) for attr in attrs}
    best_attr = max(gains, key=gains.get)
    # print(f"Chosen best attribute: {best_attr} with gain {gains[best_attr]}")

    # Create a new branch node for the best attribute
    branches = {}
    for val in X[best_attr].cat.categories: 
        # print(f"Creating branch for {best_attr} = {val}")
        # Create subset for each value of best_attr
        subset_X = X[X[best_attr] == val]
        subset_y = y[X[best_attr] == val]

        # Recursive call for the subtree
        branches[val] = learn_decision_tree(subset_X, subset_y, [a for a in attrs if a != best_attr], y)

    return DecisionBranch(best_attr, branches)


def fit(X: pd.DataFrame, y: pd.Series) -> DecisionNode:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, list(X.columns), y)


def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.
    return X.apply(lambda row: tree.predict(row), axis=1)


def load_adult(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        dtype={
            "age": int,
            "workclass": "category",
            "education": "category",
            "marital-status": "category",
            "occupation": "category",
            "relationship": "category",
            "race": "category",
            "sex": "category",
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": "category",
        },
    )
    labels = pd.read_table(label_file).squeeze().rename("label")


    # Select columns and choose a discretization for any continuous columns. Our decision tree algorithm
    # only supports discretized features and so any continuous columns (those not already "category") will need
    # to be discretized.

    # For example the following discretizes "hours-per-week" into "part-time" [0,40) hours and
    # "full-time" 40+ hours. Then returns a data table with just "education" and "hours-per-week" features.

    examples["hours-per-week"] = pd.cut(
        examples["hours-per-week"],
        bins=[0, 40, sys.maxsize],
        right=False,
        labels=["part-time", "full-time"],
    )


    examples["age"] = pd.cut(
        examples["age"],
        bins=[0, 19, 39, 59, 79, 99, sys.maxsize],
        right=False,
        labels=["0-19", "20-39", "40-59", "60-79", "80-99", "100+"],
    )

    examples["capital-gain"] = pd.cut(
        examples["capital-gain"],
        bins=[0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, sys.maxsize],
        right=False,
        labels=["0-10000", "10000-20000", "20000-30000", "30000-40000", "40000-50000", "50000-60000", "60000-70000", "70000-80000", "80000-90000", "90000-100000", "100000+"],
    )

    examples["capital-loss"] = pd.cut(
        examples["capital-loss"],
        bins=[0, 1000, 2000, 3000, 4000, 5000, sys.maxsize],
        right=False,
        labels=["0-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000", "5000+"],
    )

    return examples[["education", "hours-per-week", "sex", "occupation", "marital-status", "age"]], labels


# You should not need to modify anything below here


def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, tennis, hepatitis, adult.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )

    args = parser.parse_args()

    if args.prefix != "adult":
        # Derive input files names for test sets
        train_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_data.txt"
        )
        train_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_label.txt"
        )
        test_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_data.txt"
        )
        test_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_label.txt"
        )

        # Load training data and learn decision tree
        train_data, train_labels = load_examples(train_data_file, train_labels_file)
        tree = fit(train_data, train_labels)
        tree.display()

        # Load test data and predict labels with previously learned tree
        test_data, test_labels = load_examples(test_data_file, test_labels_file)
        pred_labels = predict(tree, test_data)

        # Compute and print accuracy metrics
        predict_metrics = compute_metrics(test_labels, pred_labels)
        for met, val in predict_metrics.items():
            print(
                met.capitalize(),
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
    else:
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.
        data_file = os.path.join(os.path.dirname(__file__), "data", "adult.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "adult.label.txt")
        data, labels = load_adult(data_file, labels_file)

        scores = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        for train_index, test_index in kfold.split(data, labels):
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            tree = fit(X_train, y_train)
            y_pred = predict(tree, X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            tree.display()

        print(
            f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(scores)} ({np.std(scores)})"
        )
