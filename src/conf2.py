import os
from pathlib import Path

def _get_current_file_dir() -> Path:
    """Returns the directory of the script."""
    try:
        return Path(os.path.realpath(__file__)).parent
    except(NameError):
        return Path(os.getcwd())


# Project root directory, i.e. the github repo directory.
_PROJECT_ROOT = _get_current_file_dir() / '..'

conf = {
        'in': {
        'labeled':   str(_PROJECT_ROOT / 'data' / 'labeledTrainData.tsv'),
        'unlabeled': str(_PROJECT_ROOT / 'data' / 'unlabeledTrainData.tsv'),
        'test':      str(_PROJECT_ROOT / 'data' / 'testData.tsv'),
        'clean':     str(_PROJECT_ROOT / 'data' / 'cleanReviews.tsv'),
    },
    'out': {
        'result':       str(_PROJECT_ROOT / 'results' / 'Prediction.csv'),
        'wrong_result': str(_PROJECT_ROOT / 'results' / 'FalsePrediction.csv'),
    },
    'vectorizer': {
        # Type of the vectorizer, one of {'word2vec', 'bagofwords'}
        'type': 'word2vec',
        'args': {},
    },
    'classifier': {
    	# Type of the classifier to use, one of {'random-forest', 'logistic-regression', 'naive-bayes'}
        # NOTE: Currently, 'random-forest' is the only working option.
	# for 'naive-bayes', activate alpha arg. 
        'type': 'random-forest',
        'args': {
        #For Naive-bayes-bagofword
            #'alpha': 0.1,  
         #For Naive-bayes-word2vec
            #'alpha': 1.2,  #0.9 for model of kaggle     
	#For logistic regression    
#	    'penalty':'l2', 
#	    'dual':True,    
#	    'tol': 0.0001,  
#	    'C':1,          
#	    'fit_intercept': True, 
#	    'intercept_scaling':1.0,
#	    'class_weight':None, 
#	    'random_state':None,
	#For Random Forest
            #'random-forest':{
                'n_estimators': 700,
                'n_jobs':       -1,
                'max_depth':    5,
                'max_features': 'auto'
            #},
        },
    },
    'run': {
        # Type of the run, one of {'optimization', 'submission'}
        'type':             'optimization',
        'number_splits':    3,
        'remove_stopwords': False,
        'cache_clean':      True,
        'test_10':          False,
        'random':           42,
        'alpha':            0.1,
    },
    'bagofwords': {},
    'word2vec': {
        # data you want to use, one of {'model', 'dictionary'}
        'data': 'dictionary', 
        'model':    str(_PROJECT_ROOT / 'results'
                                      / '300features_40minwords_10context'),
        'dictionary': str(_PROJECT_ROOT / 'dictionary_pretrained.npy'),
        'retrain':  False,
        # Averaging strategy to use, one of {'average', 'k-means'}
        'strategy': 'average'
    },
    'average': {},
    'k-means': {
        'number_clusters_frac': 0.2,  # NOTE: This argument is required!
        'max_iter':             100,
        'n_jobs':               2,
    },
}
    
