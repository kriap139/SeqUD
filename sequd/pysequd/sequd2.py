import time
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from matplotlib import pylab as plt
from sklearn.model_selection import cross_val_score
from typing import Iterable
import math
import pyunidoe as pydoe
from sequd import SeqUD, MappingData
from itertools import chain



class SeqUD2(SeqUD):
    def __init__(self, para_space, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=True, random_state=0, verbose=0, include_cv_folds=True, 
                 approx_method='linear', t=0.1):
        super().__init__(para_space, n_runs_per_stage, max_runs, max_search_iter, n_jobs, estimator, cv, scoring, refit, random_state, verbose, include_cv_folds)
        self.adjusted_ud_names = [f"{name}_UD_adjusted" for name in self.para_names]
        self.max_score_column_name = "max_prev_score"

        self.t = t
        self.approx_method = approx_method
    
    def _generate_init_design(self):
        self._adjusted_ud_logs = pd.DataFrame()
        return super()._generate_init_design()
    
    def _get_prev_stage_rows(self) -> pd.DataFrame:
        logs = self.logs
        stage = clf.logs["stage"].max() # prev stage
        stage_rows = clf.logs[clf.logs["stage"] == stage] # last stage trials
        return stage_rows

    def _para_mapping(self, para_set_ud, log_append=True):
        if not len(self.logs):
            return super()._para_mapping(para_set_ud)

        stage_rows = self._get_prev_stage_rows()
        max_score = stage_rows["score"].max() # max score of last stage

        max_rows = stage_rows[stage_rows["score"] == max_score]
        if max_rows.shape[0] > 1:
            # maybe do something if there are duplicates of the max score?
            pass
        
        center = max_rows.iloc[0]
        center_ud = center[self.para_ud_names]
        set_vecs = center_ud - para_set_ud
        transformed: pd.DataFrame = para_set_ud + self.t * set_vecs

        mapping_data = super()._para_mapping(transformed)

        if log_append:
            transformed.columns = [f"{name}_adjusted" for name in transformed.columns]
            log_aug = transformed.to_dict()
            log_aug[self.max_score_column_name] = max_score
            log_aug = pd.DataFrame(log_aug)
            mapping_data.logs_append = log_aug

        return mapping_data
    
    def _run(self, obj_func):
        super()._run(obj_func)
        columns = list(chain.from_iterable([self.para_ud_names, self.adjusted_ud_names, [self.max_score_column_name], self.para_names]))
        columns.extend(col for col in self.logs.columns if col not in columns)
        self.logs = self.logs[columns]

if __name__ == "__main__":
    from sklearn import svm
    from sklearn import datasets
    from sklearn.model_selection import KFold 
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import make_scorer, accuracy_score

    sx = MinMaxScaler()
    dt = datasets.load_breast_cancer()
    x = sx.fit_transform(dt.data)
    y = dt.target

    ParaSpace = {'C':     {'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2}, 
                'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}

    estimator = svm.SVC()
    score_metric = make_scorer(accuracy_score)
    cv = KFold(n_splits=5, random_state=0, shuffle=True)

    clf = SeqUD(ParaSpace, n_runs_per_stage=20, n_jobs=1, estimator=estimator, cv=cv, scoring=score_metric, refit=True, verbose=2, include_cv_folds=False)
    clf.fit(x, y)

    clf2 = SeqUD2(ParaSpace, n_runs_per_stage=20, n_jobs=1, estimator=estimator, cv=cv, scoring=score_metric, refit=True, verbose=2, include_cv_folds=False)
    clf2.fit(x, y)
    print(clf2.logs.tail())

    print(f"SeqUD: {clf.best_score_}, SeqUD2: {clf2.best_score_}")

