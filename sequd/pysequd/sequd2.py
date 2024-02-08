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
from itertools import chain

# Cirular imports fix
if __name__ == "__main__":
    from sequd import SeqUD, MappingData
else:
    from .sequd import SeqUD, MappingData
    __all__ = ["SeqUD2"]

class SeqUD2(SeqUD):
    def __init__(self, para_space, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=True, random_state=0, verbose=0, include_cv_folds=True, 
                 adjust_method='linear', t=0.1, exp_step = 0.1):
        super().__init__(para_space, n_runs_per_stage, max_runs, max_search_iter, n_jobs, estimator, cv, scoring, refit, random_state, verbose, include_cv_folds)
        self.adjusted_ud_names = [f"{name}_UD_adjusted" for name in self.para_names]
        self.max_score_column_name = "max_prev_score"

        self.t = t
        self.k = 12
        self.x_0 = 0.5
        self.exp_step = exp_step
        self.adjust_method = adjust_method

        if adjust_method == "linear":
            self.adjust = self.adjust_linear
        elif adjust_method == "exp":
            self.adjust = self.adjust_exp
        else:
            raise ValueError(f"Invalid adjust method ({adjust_method}). Supported values are ('linear', 'exp')")
    
    def adjust_linear(self, set_vecs: pd.DataFrame) -> pd.DataFrame:
        return self.t * set_vecs
    
    def sigmoid(self, x: float) -> float: 
        return 1 / (1 + np.exp(-self.k * (x - self.x_0)))

    def adjust_exp(self, set_vecs):
        frac = self.sigmoid(self.exp_step * (self.stage - 1)) 
        return frac * set_vecs

    def _generate_init_design(self):
        self._adjusted_ud_logs = pd.DataFrame()
        return super()._generate_init_design()
    
    def _get_prev_stage_rows(self) -> pd.DataFrame:
        # self.logs to clf in stage and stage_rows for the increased score bug.
        stage = self.logs["stage"].max() # prev stage
        stage_rows = self.logs[self.logs["stage"] == stage] # last stage trials
        return stage_rows

    def _para_mapping(self, para_set_ud, log_append=True):
        # Initial Stage
        if not len(self.logs):
            return super()._para_mapping(para_set_ud)

        stage_rows = self._get_prev_stage_rows()
        max_score = stage_rows["score"].max() # max score of last stage

        max_rows = stage_rows[stage_rows["score"] == max_score]
        if max_rows.shape[0] > 1:
            # maybe do something if there are duplicates of the max score?
            pass
        
        center = max_rows.iloc[0]
        if self.stage == 2:
            # Previous stage was the initial stage, so no adjustments to the search space have been made.
            center_ud = center[self.para_ud_names]
        else:
            center_ud = center[self.adjusted_ud_names]
            center_ud.rename({col: col.rstrip("_adjusted") for col in center_ud.index}, inplace=True)

        set_vecs = center_ud - para_set_ud
        transformed: pd.DataFrame = para_set_ud + self.adjust(set_vecs)

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
        # Sorting columns
        columns = list(chain.from_iterable([self.para_ud_names, self.adjusted_ud_names, [self.max_score_column_name], self.para_names]))
        columns.extend(col for col in self.logs.columns if col not in columns)

        # The search didn't go past the initial stage, so no adjusted UD columns have been added!
        if self.stage - 1 == 1:
            added_cols = list(chain.from_iterable([self.adjusted_ud_names, [self.max_score_column_name]]))
            self.logs[added_cols] = np.nan
        
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

    #clf = SeqUD(ParaSpace, n_runs_per_stage=20, n_jobs=1, estimator=estimator, cv=cv, 
    #    scoring=score_metric, refit=True, verbose=2, include_cv_folds=False
    #)
    #clf.fit(x, y)

    clf2 = SeqUD2(ParaSpace, n_runs_per_stage=20, n_jobs=1, estimator=estimator, cv=cv, 
        scoring=score_metric, refit=True, verbose=2, include_cv_folds=False, t=0.25,
        adjust_method='linear', exp_step=0.18
    )
    clf2.fit(x, y)

    # exp_step = 0.9824561403508772
    # t = 0.9824561403508772
    print(f"SeqUD: 0.9807017543859649, SeqUD2: {clf2.best_score_}")

