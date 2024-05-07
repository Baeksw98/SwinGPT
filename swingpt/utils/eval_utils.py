from nltk import word_tokenize, download
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import StoppingCriteria, StoppingCriteriaList
import copy
import numpy as np
import torch
from collections import defaultdict, Counter
from torchmetrics import Metric

class PunctuationStoppingCriteria(StoppingCriteria):
    """
    Stop generation when any of the specified punctuation marks is generated.
    """
    def __init__(self, punctuation_marks, tokenizer):
        self.punctuation_marks = punctuation_marks
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        last_token_id = input_ids[0, -1].item()  
        last_token = self.tokenizer.decode([last_token_id])
        return last_token in self.punctuation_marks
    
def ensure_nltk_resources():
    """Ensure necessary NLTK resources are downloaded."""
    try:
        # Try tokenizing a dummy text to check if 'punkt' is downloaded
        word_tokenize("Test text.")
    except LookupError:
        # If not found, download 'punkt'
        download('punkt')
        
def generate_ngrams(sentence, n=2):
    """Generate n-grams from a given sentence."""
    # Ensure the punkt tokenizer is downloaded
    ensure_nltk_resources()

    # Generate up to n-grams
    tokens = word_tokenize(sentence)
    return [list(ngrams(tokens, i)) for i in range(1, n+1)]

class BLEUscore(Metric):
    """ Custom Metric class for computing BLEU (Bilingual Evaluation Understudy) scores. """

    def __init__(self, weights=(0.25, 0.25, 0.25, 0.25), dist_sync_on_step=False):
        """ Initializes the BLEU score metric. """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.weights = weights
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """ Update the state with predicted and target data. """
        # Ensure necessary NLTK resources are ready
        ensure_nltk_resources()

        # Prepare the reference captions and hypothesis captions
        tokenized_refs = [word_tokenize(ref.lower()) for ref in targets]
        
        # Prepare the hypothesis captions
        if isinstance(preds, str):
            preds = [preds]  # Convert single string to list
        tokenized_preds = [word_tokenize(pred.lower()) for pred in preds]
        
        # Calculate BLEU scores for each hypothesis
        scores = [sentence_bleu(tokenized_refs, pred, weights=self.weights, smoothing_function=SmoothingFunction().method1) for pred in tokenized_preds]
        
        # Sum scores and increment count
        score_sum = torch.tensor(scores).sum().float()
        self.score_sum += score_sum
        self.total += len(preds)
        
    def compute(self):
        """ Calculate the final BLEU score over all updated data. """
        return self.score_sum / self.total if self.total != 0 else torch.tensor(0.0)
        
def prepare_captions(captions):
    """Prepare and tokenize captions."""
    ensure_nltk_resources()
    # Tokenize captions
    return [word_tokenize(cap.lower()) for cap in captions]

class CIDERscore(Metric):
    """ Custom Metric class for computing CIDEr (Consensus-based Image Description Evaluation) scores. """

    def __init__(self, n=4, sigma=6.0, dist_sync_on_step=False):
        """ Initializes the CIDEr score metric. """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n = n
        self.sigma = sigma
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """ Update the state with predicted and target data. """
        
        # Wrap each target in a list
        formatted_targets = [[t] for t in targets]  
        cider_scorer = CiderScorer(n=self.n, sigma=self.sigma)
        
        for pred, target_group in zip(preds, formatted_targets):
            # Each target_group is a list containing a single reference sentence
            cider_scorer.cook_append(pred, target_group)
        score, _ = cider_scorer.compute_score()
        
        # Ensure tensor is of type float
        self.score_sum += torch.tensor(score).float()  
        self.total += 1

    def compute(self):
        """ Calculate the final CIDEr score over all updated data. """
        return self.score_sum / self.total if self.total != 0 else torch.tensor(0.0)
    
    
class CiderScorer(object):
    """ Utility class for calculating CIDEr scores for a set of predictions and references. """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        """ Initializes the CiderScorer. """
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.ref_len = None
        if refs is not None and test is not None:
            self.cook_append(test, refs)

    def cook_append(self, test, refs):
        """ Appends a new set of test and reference data to the scorer. """
        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
        if test is not None:
            self.ctest.append(precook(test, self.n))
        
    def compute_doc_freq(self):
        """ Calculates the document frequency of each n-gram across all reference sentences. """
        for refs in self.crefs:
            for ref in refs:
                for ngram, count in ref.items():
                    self.document_frequency[ngram] += 1
        # Setting ref_len properly after all document frequencies are computed
        if len(self.crefs) > 0:
            self.ref_len = np.log(float(len(self.crefs)))
        else:
            # Fallback to avoid log(0) if no references are added
            self.ref_len = np.log(1.0)  

    def compute_score(self, option=None, verbose=0):
        """ Compute the overall CIDEr score for all appended test and reference pairs. """
        self.compute_doc_freq()
        assert self.ref_len is not None, "Reference length not set. Ensure document frequencies are computed."
        scores = self.compute_cider()
        return np.mean(scores), scores
    
    def compute_cider(self):
        """ Computes CIDEr scores for each individual test and reference pair. """
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = self.counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = self.counts2vec(ref)
                score += self.sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score / len(refs)) * 10.0
            scores.append(score_avg)
        return scores

    def counts2vec(self, cnts):
        """ Converts count of n-grams into a vector form using TF-IDF weighting. """
        vec = [defaultdict(float) for _ in range(self.n)]
        norm = [0.0 for _ in range(self.n)]
        length = 0
        for ngram, count in cnts.items():
            df = np.log(max(1.0, self.document_frequency[ngram]))
            n = len(ngram) - 1
            tf_idf = count * (self.ref_len - df)
            vec[n][ngram] = tf_idf
            norm[n] += tf_idf ** 2
            if n == 1:
                length += count
        norm = [np.sqrt(n) for n in norm]
        return vec, norm, length

    def sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        """ Computes the similarity between the vector representation of hypothesis and references. """
        delta = length_hyp - length_ref
        val = np.array([0.0 for _ in range(self.n)])
        for n in range(self.n):
            for ngram, count in vec_hyp[n].items():
                val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
            if norm_hyp[n] and norm_ref[n]:
                val[n] /= (norm_hyp[n] * norm_ref[n])
            val[n] *= np.exp(-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    def copy(self):
        """ Creates a copy of this CiderScorer instance. """
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new
    
def cook_refs(refs, n=4):
    """ Preprocesses a list of reference sentences into n-gram counts. """
    return [precook(ref, n) for ref in refs]

def precook(s, n=4, out=False):
    """ Converts a single sentence into n-gram counts. """
    words = s.split()
    counts = Counter()
    for k in range(1, n+1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_test(test, n=4):
    """ Prepares a test sentence for evaluation by converting it into n-gram counts. """
    return precook(test, n, True)

    