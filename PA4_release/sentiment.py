"""
CS311 Programming Assignment 4: Naive Bayes

Full Name: Sujay Banerjee

Brief description of my custom classifier:
I added n-grams from 2-5, document length, number of capitalization, and number of punctuation(?, !) to my features. 

TODO: Answer the questions included in the assignment
"""
import argparse, math, os, re, string, zipfile
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
import numpy as np
from sklearn import metrics



class Sentiment:
    """Naive Bayes model for predicting text sentiment"""

    def __init__(self, labels: Iterable[Hashable]):
        """Create a new sentiment model

        Args:
            labels (Iterable[Hashable]): Iterable of potential labels in sorted order.

        Attributes:
            sentiment_label: List of sentiment labels
            document_counts: Number of documents for each sentiment S
            word_counts: Word counts for each sentiment S
            total_words: Total word counts for each sentiment S
            unique_words: Unique words for each sentiment S
            vocabulary: All unique words from either sentiment S
        """
        self.sentiment_label = labels   # 0 for negative, 1 for positive
        self.document_counts = {label: 0 for label in labels}  
        self.word_counts = {label: {} for label in labels}  
        self.total_words = {label: 0 for label in labels}  
        self.unique_words = {label: set() for label in labels}  
        self.vocabulary = set()     
        

    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Normalize the string into a list of words.

        Args:
            example (str): Text input to split and normalize
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            List[str]: Normalized words
        """
        # TODO: Modify the method to generate individual words from the example. Example modifications include
        # removing punctuation and/or normalizing case (e.g., making all lower case)

        example = example.lower() # convert to lower case
        example = example.translate(str.maketrans('', '', string.punctuation)) # remove punctuation, copilot
        return example.split()

    def add_example(self, example: str, label: Hashable, id:str = None):
        """Add a single training example with label to the model 

        Args:
            example (str): Text input
            label (Hashable): Example label
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        
        """
        # TODO: Implement function to update the model with words identified in this training example
    
        words = self.preprocess(example, id=id) #get the processed words
        self.document_counts[label] += 1 
        self.total_words[label] += len(words)

        for word in words:
            self.vocabulary.add(word)
            self.unique_words[label].add(word)
            if word in self.word_counts[label]:
                self.word_counts[label][word] += 1
            else:
                self.word_counts[label][word] = 1


    def predict(self, example: str, pseudo=0.0001, id:str = None) -> Sequence[float]:
        """Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels
        """

        words = self.preprocess(example, id=id)   #get the processed words
        log_probs = {}  # for log space
        total_docs = sum(self.document_counts.values())  

        # compute the log probability for each label
        for label in self.sentiment_label:
            log_prior = math.log(self.document_counts[label] / total_docs) # log P(S=label)
            log_prob = log_prior # so dont accumulate log_prior with total prob
            U_S = len(self.unique_words[label])   # Number of unique words for label
            denominator = self.total_words[label] + pseudo * U_S # Denominator for P(word|label): total_words[label] + pseudo * U_S

            # For each word in the example, calculate log P(word|S)
            for word in words:
                word_count = self.word_counts[label].get(word, 0) # Get the word count for this label, N_S(word)
                numerator = word_count + pseudo # Numerator for P(word|label): N_S(word) + pseudo
                
                log_prob += math.log(numerator / denominator) # log P(word|label) 

            log_probs[label] = log_prob

        # calculate the denominator P(f) in log-space 
        log_prob_positive = log_probs[1] # log P(S=positive | f)
        log_prob_negative = log_probs[0] # log P(S=negative | f)
        log_prob_f = np.logaddexp(log_prob_positive, log_prob_negative)

        # compute log P(S|f) for each sentiment
        log_S_positive = log_prob_positive - log_prob_f
        log_S_negative = log_prob_negative - log_prob_f

        # Exponentiate to get P(S|f)
        prob_positive = np.exp(log_S_positive)
        prob_negative = np.exp(log_S_negative)

        return [prob_negative, prob_positive]  





class CustomSentiment(Sentiment):
    # TODO: Implement your custom Naive Bayes model
    def __init__(self, labels: Iterable[Hashable], n_min: int = 2, n_max = 5):
        super().__init__(labels)
        self.n_min = n_min 
        self.n_max = n_max
    
    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Normalize the string into a list of words and n-grams.

        Args:
            example (str): Text input to split and normalize
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            List[str]: List of unigrams and n-grams
        """
        # TODO: Modify the method to generate individual words from the example. Example modifications include
        # removing punctuation and/or normalizing case (e.g., making all lower case)

        text_original = example
        example = example.lower() # convert to lower case
        example = example.translate(str.maketrans('', '', string.punctuation)) # remove punctuation, copilot
        words = example.split()
        # Make n-grams
        ngrams_list = []
        for word in words:
            if len(word) < self.n_min: 
                ngrams_list.append(word)
                continue
            for n in range(self.n_min, self.n_max +1):
                for j in range(len(word) - n + 1):
                        ngram = word[j:j + n]
                        ngrams_list.append(ngram)
        features = ngrams_list #to add more features

        # document length
        doc_length = f"doc_length_{len(words)}"
        features.append(doc_length)

        # capitalization 
        capitalized_words = len([word for word in text_original.split() if word[0].isupper()])
        num_cap = f"capitalization_{capitalized_words}"
        features.append(num_cap)

        # punctuation !, ?
        exclamations = len(re.findall(r'!', text_original))
        question_marks = len(re.findall(r'\?', text_original))
        features.append(f"exclamations_{exclamations}")
        features.append(f"question_marks_{question_marks}")

        return features

    
    def add_example(self, example: str, label: Hashable, id:str = None):
        """Add a single training example with label to the model 

        Args:
            example (str): Text input
            label (Hashable): Example label
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        
        """
        # TODO: Implement function to update the model with words identified in this training example
    
        features = self.preprocess(example, id=id) #get n-grams, replace words with features
        self.document_counts[label] += 1 
        self.total_words[label] += len(features)
        

        for feature in features:
            self.vocabulary.add(feature)
            self.unique_words[label].add(feature)
            if feature in self.word_counts[label]:
                self.word_counts[label][feature] += 1
            else:
                self.word_counts[label][feature] = 1

    def predict(self, example: str, pseudo=0.001, id:str = None) -> Sequence[float]:
        """Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels
        """

        features = self.preprocess(example, id=id)
        log_probs = {}
        total_docs = sum(self.document_counts.values())

        
        # compute the log probability for each label
        for label in self.sentiment_label:
            log_prior = math.log(self.document_counts[label] / total_docs) # log P(S=label)
            log_prob = log_prior # so dont accumulate log_prior with total prob
            
            U_S = len(self.unique_words[label])   # Number of unique words for label
            
            denominator = self.total_words[label] + pseudo * U_S # Denominator for P(word|label): total_words[label] + pseudo * U_S

            # For each word in the example, calculate log P(word|S)
            for feature in features:
                feature_count = self.word_counts[label].get(feature, 0) 
                numerator = feature_count + pseudo # Numerator for P(word|label): N_S(word) + pseudo

                log_prob += math.log(numerator / denominator) # log P(word|label) 
                

            log_probs[label] = log_prob
           
        # calculate the denominator P(f) in log-space 
        log_prob_positive = log_probs[1] # log P(S=positive | f)
        log_prob_negative = log_probs[0] # log P(S=negative | f)
        log_prob_f = np.logaddexp(log_prob_positive, log_prob_negative)


        # compute log P(S|f) for each sentiment
        log_S_positive = log_prob_positive - log_prob_f
        log_S_negative = log_prob_negative - log_prob_f

        # Exponentiate to get P(S|f)
        prob_positive = np.exp(log_S_positive)
        prob_negative = np.exp(log_S_negative)

        return [prob_negative, prob_positive]    


def process_zipfile(filename: str) -> Generator[Tuple[str, str, int], None, None]:
    """Create generator of labeled examples from a Zip file that yields a tuple with
    the id (filename of input), text snippet and label (0 or 1 for negative and positive respectively).

    You can use the generator as a loop sequence, e.g.

    for id, example, label in process_zipfile("test.zip"):
        # Do something with example and label

    Args:
        filename (str): Name of zip file to extract examples from

    Yields:
        Generator[Tuple[str, str, int], None, None]: Tuple of (id, example, label)
    """
    with zipfile.ZipFile(filename) as zip:
        for info in zip.infolist():
            # Iterate through all file entries in the zip file, picking out just those with specific ratings
            match = re.fullmatch(r"[^-]+-(\d)-\d+.txt", os.path.basename(info.filename))
            if not match or (match[1] != "1" and match[1] != "5"):
                # Ignore all but 1 or 5 ratings
                continue
            # Extract just the relevant file the Zip archive and yield a tuple
            with zip.open(info.filename) as file:
                yield (
                    match[0],
                    file.read().decode("utf-8", "ignore"),
                    1 if match[1] == "5" else 0,
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
    model = CustomSentiment([0, 1])
    model.add_example("good", 1)
    model.add_example("bad", 0)

    prediction = model.predict("good", 0.0001)
    print(prediction)


    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment analyzer")

    parser.add_argument(
        "--train",
        default="data/train.zip",
        help="Path to zip file or directory containing training files.",
    )
    parser.add_argument(
        "--test",
        default="data/test.zip",
        help="Path to zip file or directory containing testing files.",
    )
    parser.add_argument(
        "-m", "--model", default="base", help="Model to use: One of base or custom"
    )
    parser.add_argument("example", nargs="?", default=None)

    args = parser.parse_args()

    # Train model
    if args.model == "custom":
        model = CustomSentiment(labels=[0, 1])
    else:
        model = Sentiment(labels=[0, 1])
    for id, example, y_true in process_zipfile(
        os.path.join(os.path.dirname(__file__), args.train)
    ):
        model.add_example(example, y_true, id=id)

    # If interactive example provided, compute sentiment for that example
    if args.example:
        print(model.predict(args.example))
    else:
        predictions = []
        for id, example, y_true in process_zipfile(
            os.path.join(os.path.dirname(__file__), args.test)
        ):
            # Determine the most likely class from predicted probabilities
            predictions.append((id, y_true, np.argmax(model.predict(example,id=id))))

        # Compute and print accuracy metrics
        _, y_test, y_true = zip(*predictions)
        predict_metrics = compute_metrics(y_test, y_true)
        for met, val in predict_metrics.items():
            print(
                f"{met.capitalize()}: ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )

