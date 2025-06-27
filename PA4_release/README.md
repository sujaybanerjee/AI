# CS311 Programming Assignment 4: Naïve Bayes

For this assignment, you will be training and testing Naïve Bayes models for predicting text sentiment. Refer to the Canvas assignment for assignment specifications. This README describes how to run the skeleton code.

## Running the skeleton program

The skeleton code trains and tests your Naïve Bayes model on the provide training and test data and can also be used to predict the sentiment for provided text. Executing `sentiment.py` will train and test your model by default, e.g.,

```
$ python3 sentiment.py 
Confusion: 
[[10751  2436]
 [ 4293  9707]]
Accuracy: 0.7524919998528709
Recall: 0.6933571428571429
Precision: 0.79939059540476
F1: 0.7426079638909078
```

You can provide your own string as an optional argument.

```
$ python3 sentiment.py -h
usage: sentiment.py [-h] [--train TRAIN] [--test TEST] [-m MODEL] [example]

Train Naive Bayes sentiment analyzer

positional arguments:
  example

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Path to zip file or directory containing training files.
  --test TEST           Path to zip file or directory containing testing files.
  -m MODEL, --model MODEL
                        Model to use: One of base or custom
```

For example, to test your program on a specific string you can provide it as positional argument, e.g., `python3 sentiment.py "computer science is awesome"`. Use double quotes to ensure that your input string is treated as a single input.

```
$ python3 sentiment.py "computer science is awesome"
[0.23071661 0.76928339]
```

If you are working with Thonny, recall that you can change the command line arguments by modifying the `%Run` command in the shell, e.g., `%Run sentiment.py "computer science is awesome"`.

** Note that the skeleton code will read the the zip file directly. You do not need to unzip those files.** 

## Unit testing

To assist you during development, a unit test suite is provided in `sentiment_test.py`. These tests are a subset of the tests run by Gradescope. You can run the tests by executing the `sentiment_test.py` file as a program, e.g. `python3 sentiment_test.py`. 

```
$ python3 sentiment_test.py
```