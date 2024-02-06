import unittest
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

from sequd import SeqUD
from sequd import SeqUD2
#from sequd import GPEIOPT
#from sequd import SMACOPT
#from sequd import TPEOPT

sx = MinMaxScaler()
dt = datasets.load_breast_cancer()
x = sx.fit_transform(dt.data)
y = dt.target

ParaSpace = {'C': {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2},
             'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

estimator = svm.SVC()
score_metric = make_scorer(accuracy_score, greater_is_better=True)
cv = KFold(n_splits=5, random_state=0, shuffle=True)


class TestSeqUD(unittest.TestCase):
    """
    SeqUD test class.
    """
    def setUp(self):
        """ Initialise test suite - no-op. """
        pass

    def tearDown(self):
        """ Clean up test suite - no-op. """
        pass

    def test_SeqUD(self):
        """ Test SeqUD. """

        clf = SeqUD(ParaSpace, n_runs_per_stage=20, max_runs=10, estimator=estimator, cv=cv,
                    scoring=score_metric, n_jobs=1, refit=True, verbose=False, include_cv_folds=False)
        clf.fit(x, y)
    
    def test_SeqUD2(self):
        clf = SeqUD2(ParaSpace, n_runs_per_stage=20, max_runs=10, estimator=estimator, cv=cv,
                    scoring=score_metric, n_jobs=1, refit=True, verbose=0, include_cv_folds=False)
        clf.fit(x, y)

    # def test_GPEI(self):
    #     """ Test GPEI. """
    #     clf = GPEIOPT(ParaSpace, max_runs=10, time_out=10, estimator=estimator, cv=cv, refit=True, verbose=False)
    #     clf.fit(x, y)

    # def test_SMAC(self):
    #     """ Test SMAC. """
    #     clf = SMACOPT(ParaSpace, max_runs=10, estimator=estimator, cv=cv, refit=True, verbose=False)
    #     clf.fit(x, y)

    # def test_TPE(self):
    #     """ Test TPE. """
    #     clf = TPEOPT(ParaSpace, max_runs=10, estimator=estimator, cv=cv, refit=True, verbose=False)
    #     clf.fit(x, y)


if __name__ == '__main__':
    unittest.main()
