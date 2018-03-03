import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        bestScore = float("inf")
        bestModel = None

        for numberOfComponents in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(numberOfComponents)

                logLikelihood = model.score(self.X, self.lengths)
                logN = np.log(self.X.shape[0])
                numberOfParams = (numberOfComponents * numberOfComponents) + (2 * numberOfComponents * self.X.shape[1]) -1

                bic = -2 * logLikelihood + numberOfParams * logN

                if bic < bestScore:
                    bestScore = bic
                    bestModel = model

            except Exception as e:
                continue

        if bestModel is not None:
            return bestModel  
        else:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def logLikelihoodOfOtherWords(self, model, otherWordLengthPairs):
        return [model[0].score(wordLengthPair[0], wordLengthPair[1]) for wordLengthPair in otherWordLengthPairs]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores
        
        # Grab every word except this one
        otherWordLengthPairs = [self.hwords[word] for word in self.words if word != self.this_word]
        # (model, log)
        models = []

        bestScore = float("-inf")
        bestModel = None

        for numberOfComponents in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(numberOfComponents)
                logLikelihood = model.score(self.X, self.lengths)
                models.append((model, logLikelihood))

            except Exception as e:
                continue
        
        for index, model in enumerate(models):
            hmmModel, logLikelihoodOfOriginalWord = model
            score = logLikelihoodOfOriginalWord - np.mean(self.logLikelihoodOfOtherWords(model, otherWordLengthPairs))

            if score > bestScore:
                bestScore = score
                bestModel = hmmModel


        if bestModel is not None:
            return bestModel  
        else:
            return self.base_model(self.n_constant)




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        bestScore = float("-inf")
        bestModel = None
        average = 0
        
        for numberOfComponents in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            numberOfSplits = 3
            model = None
            
            if(len(self.sequences) < numberOfSplits):
                break
            
            kFolds = KFold(random_state=self.random_state, n_splits=numberOfSplits)

            for cvTrainingIndex, cvTestingIndex in kFolds.split(self.sequences):
                trainingX, trainingLengths = combine_sequences(cvTrainingIndex, self.sequences)
                testingX,  testingLengths  = combine_sequences(cvTestingIndex, self.sequences)

                try:
                    hmm = GaussianHMM(n_components=numberOfComponents, n_iter=1000, random_state=inst.random_state)
                    model = hmm.fit(trainingX, trainingLengths)

                    scores.append(model.score(testingX, testingLengths))
                    
                except Exception as e:
                    break

            if len(scores) > 0:
                average = np.average(scores)  
            else: 
                average = float("-inf")
            
            if average > bestScore:
                bestScore = average
                bestModel = model
        
        if bestModel is not None:
            return bestModel  
        else:
            return self.base_model(self.n_constant)

        
