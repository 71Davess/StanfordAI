#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass
    # ### START CODE HERE ###
    
    # 1.Creation of the empty feature vector 'review_vector'
    # 2.Split the input string at empty spaces
    # 3.Iterate the dictionary, checking if the word is there
    # If not, the word is introduced and the value is updated.
    # If it is, only the value is updated (done with dict.get function)

    review_phi = {}
    words = x.split()

    for w in words:
        review_phi[w] = review_phi.get(w,0) + 1
       
    return (review_phi)
    
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight
    # ### START CODE HERE ###

    # If the Score (phi(x) . w) is exactly 0, output +1. Otherwise, output -1.
    def predictor(x: T) -> int:
        return 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1
    # Stochastic Gradient Descent:
        #First, iterate over the number of epochs.(1....T or numEpochs)
        #Then, iterate over each training example (x,y).
        #After phi, the margin is computed. 
    for epoch in range(numEpochs):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            margin = dotProduct(weights, phi) * y
            # If margin is less than 1, loss hinge is greater than 0, thereby not converged. So, update weights.
            if margin < 1:  
                # Update weights according to the formula.
                increment(weights, eta * y, phi)

        # Monitor training progress.
        trainError = evaluatePredictor(trainExamples, predictor)
        validationError = evaluatePredictor(validationExamples, predictor)
        print(f"Epoch {epoch}: train error = {trainError}, validation error = {validationError}")

    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None 
        y = None
        # ### START CODE HERE ###
        phi = {} #Empty feature vector as a dictionary
        keys = list(weights.keys())

        if keys:
            # choose a non-empty subset of features randomly
            subset_size = random.randint(1, len(keys))
            chosen_keys = random.sample(keys, subset_size)
            for k in chosen_keys:
                # random non-zero value to allow positive/negative scores
                phi[k] = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])

            score = dotProduct(phi, weights)

            """
            #---------------------SCORE DEBUGGING------------------------
            #Even though the phi and weights are non-zero, some errors
            # may lead to zero score. This block is to debug such cases.
            print("Score =", score)
            # avoid zero score so label is well-defined
            #check_zero_weights = 0 in weights.values()
            #print(check_zero_weights)
            if score == 0:
                print("###############SCORE IS ZERO#################")
                k0 = chosen_keys[0]
                phi[k0] += 1 if weights[k0] >= 0 else -1
                score = dotProduct(phi, weights)
            #-----------------------------------------------------------
            """

            y = 1 if score >= 0 else -1
        else:
            # no weights: return empty features with positive label
            phi = {}
            y = 1
        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        # ### START CODE HERE ###
        phi = {}
        s = x.replace(" ", "") # Remove spaces from the string
        # Iterate over the string to extract n-grams avoiding index errors
        #The 'gram' variable stores the n-gram substring on phi update.
        for i in range(len(s) - n + 1):
            gram = s[i : i + n]
            phi[gram] = phi.get(gram, 0) + 1
        return phi
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from submission import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # ### START CODE HERE ###
    # ### END CODE HERE ###
