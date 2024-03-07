import unittest
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

from sequd import SeqUD
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
                    scoring=score_metric, n_jobs=1, refit=True, verbose=False)
        clf.fit(x, y)
        # originall prallel setup commit: 4d2eb6075c6a95cdc8a65e32a1d06823ab41385c

        # Results_new_parallel_setup: best_score=0.9789473684210528, best_params={'C': 46.85074227025992, 'gamma': 0.45062523130541443}
        print(f"best_score={clf.best_score_}, best_params={clf.best_params_}")

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
