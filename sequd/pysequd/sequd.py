import time
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from matplotlib import pylab as plt
from sklearn.model_selection import cross_val_score
from typing import Iterable
import math
from dataclasses import dataclass
import pyunidoe as pydoe

EPS = 10**(-8)

@dataclass
class MappingData:
    para_set: pd.DataFrame
    logs_append: pd.DataFrame

def time_to_str(secs: float) -> str:
    secs = int(secs)
    days, r = divmod(secs, 86400)
    hours, r = divmod(r, 3600)
    minutes, secs = divmod(r, 60)
    if days == 0:
        return "{:02}:{:02}:{:02}".format(hours, minutes, secs)
    else:
        return "{:02}:{:02}:{:02}:{:02}".format(days, hours, minutes, secs)

class SeqUD(object):

    """
    Implementation of sequential uniform design.

    Parameters
    ----------
    :type  para_space: dict or list of dictionaries
    :param para_space: It has three types:

        Continuous:
            Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
            `Wrapper`, a callable function for wrapping the values.
        Integer:
            Specify `Type` as `integer`, and include the keys of `Mapping` (a list with all the sortted integer elements).
        Categorical:
            Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories).

    :type n_runs_per_stage: int, optional, default=20
    :param n_runs_per_stage: The positive integer which represent the number of levels in generating uniform design.

    :type max_runs: int, optional, default=100
    :param max_runs: The maximum number of trials to be evaluated. When this values is reached,
        then the algorithm will stop.

    :type max_search_iter: int, optional, default=100
    :param max_search_iter: The maximum number of iterations used to generate uniform design or augmented uniform design.

    :type n_jobs: int or None, optional, optional, default=None
    :param n_jobs: Number of jobs to run in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code
        is used at all, which is useful for debugging. See the package `joblib` for details.

    :type  estimator: estimator object
    :param estimator: This is assumed to implement the scikit-learn estimator interface.

    :type  cv: cross-validation method, an sklearn object.
    :param cv: e.g., `StratifiedKFold` and KFold` is used.

    :type scoring: string, callable, list/tuple, dict or None, optional, default=None
    :param scoring: A sklearn type scoring function.
        If None, the estimator's default scorer (if available) is used. See the package `sklearn` for details.

    :type refit: boolean, or string, optional, default=True
    :param refit: It controls whether to refit an estimator using the best found parameters on the whole dataset.

    :type random_state: int, optional, default=0
    :param random_state: The random seed for optimization.

    :type verbose: boolean, optional, default=False
    :param verbose: It controls whether the searching history will be printed.


    Examples
    ----------
    >>> import numpy as np
    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> from sequd import SeqUD
    >>> from sklearn.model_selection import KFold
    >>> iris = datasets.load_iris()
    >>> ParaSpace = {'C':{'Type': 'continuous', 'Range': [-6, 16], 'Wrapper': np.exp2},
               'gamma': {'Type': 'continuous', 'Range': [-16, 6], 'Wrapper': np.exp2}}
    >>> estimator = svm.SVC()
    >>> cv = KFold(n_splits=5, random_state=1, shuffle=True)
    >>> clf = SeqUD(ParaSpace, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=None, random_state=0, verbose=False)
    >>> clf.fit(iris.data, iris.target)

    Attributes
    ----------
    :vartype best_score\_: float
    :ivar best_score\_: The best average cv score among the evaluated trials.

    :vartype best_params\_: dict
    :ivar best_params\_: Parameters that reaches `best_score_`.

    :vartype best_estimator\_: sklearn estimator
    :ivar best_estimator\_: The estimator refitted based on the `best_params_`.
        Not available if estimator = None or `refit=False`.

    :vartype search_time_consumed\_: float
    :ivar search_time_consumed\_: Seconds used for whole searching procedure.

    :vartype refit_time\_: float
    :ivar refit_time\_: Seconds used for refitting the best model on the whole dataset.
        Not available if estimator=None or `refit=False`.
    """

    def __init__(self, para_space, n_runs_per_stage=20, max_runs=100, max_search_iter=100, n_jobs=None,
                 estimator=None, cv=None, scoring=None, refit=True, random_state=0, verbose=0, include_cv_folds=True):

        self.para_space = para_space
        self.n_runs_per_stage = n_runs_per_stage
        self.max_runs = max_runs
        self.max_search_iter = max_search_iter
        self.n_jobs = n_jobs if isinstance(n_jobs, int) else 1
        self.random_state = random_state
        self.verbose = verbose
        self.include_cv_folds = include_cv_folds

        self.cv = cv
        self.refit = refit
        self.scoring = scoring
        self.estimator = estimator

        self.stage = 0
        self.stop_flag = False
        self.para_ud_names = []
        self.variable_number = [0]
        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        for items, values in self.para_space.items():
            if (values['Type'] == "categorical"):
                self.variable_number.append(len(values['Mapping']))
                self.para_ud_names.extend(
                    [items + "_UD_" + str(i + 1) for i in range(len(values['Mapping']))])
            else:
                self.variable_number.append(1)
                self.para_ud_names.append(items + "_UD")
        self.extend_factor_number = sum(self.variable_number)
    
    def get_n_cv_folds(self) -> int:
        if self.cv is None:
            return 5
        elif hasattr(self.cv, "get_n_splits") and callable(self.cv.get_n_splits):
            return self.cv.get_n_splits()
        elif isinstance(self.cv, Iterable):
            return len(self.cv)
        elif isinstance(self.cv, int):
            return self.cv
        else:
            raise ValueError(f"cv attr is not a supported type {self.cv}")
    
    def get_n_trials(self) -> int:
        return self.logs.shape[0] * self.get_n_cv_folds()

    def plot_scores(self):
        """
        Visualize the scores history.
        """

        if self.logs.shape[0] > 0:
            cum_best_score = self.logs["score"].cummax()
            plt.figure(figsize=(6, 4))
            plt.plot(cum_best_score)
            plt.xlabel('# of Runs')
            plt.ylabel('Best Scores')
            plt.title('The best found scores during optimization')
            plt.grid(True)
            plt.show()
        else:
            print("No available logs!")

    def _summary(self):
        """
        This function summarizes the evaluation results and makes records.

        Parameters
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point,
                and columns are used to represent variables.
        para_set: A pandas dataframe which contains the trial points in original form.
        score: A numpy vector, which contains the evaluated scores of trial points in para_set.

        """
        self.best_index_ = self.logs.loc[:, "score"].idxmax()
        self.best_params_ = {self.logs.loc[:, self.para_names].columns[j]:
                             self.logs.loc[:, self.para_names].iloc[self.best_index_, j]
                             for j in range(self.logs.loc[:, self.para_names].shape[1])}
        self.best_score_ = self.logs.loc[:, "score"].iloc[self.best_index_]
        if self.verbose > 0:
            print("SeqUD completed in %.2f seconds." %
                  self.search_time_consumed_)
            print("The best score is: %.5f." % self.best_score_)
            print("The best configurations are:")
            print("\n".join("%-20s: %s" % (k, v if self.para_space[k]['Type'] == "categorical" else round(v, 5))
                            for k, v in self.best_params_.items()))

    def _para_mapping(self, para_set_ud):
        """
        This function maps trials points in UD space ([0, 1]) to original scales.

        There are three types of variables:
          - continuous：Perform inverse Maxmin scaling for each value.
          - integer: Evenly split the UD space, and map each partition to the corresponding integer values.
          - categorical: The UD space uses one-hot encoding, and this function selects the one with the maximal value as class label.

        Parameters
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point,
                and columns are used to represent variables.

        Returns
        ----------
        para_set: The transformed variables.
        """

        para_set = pd.DataFrame(
            np.zeros((para_set_ud.shape[0], self.factor_number)), columns=self.para_names)
        for item, values in self.para_space.items():
            if (values['Type'] == "continuous"):
                para_set[item] = values['Wrapper'](
                    para_set_ud[item + "_UD"] * (values['Range'][1] - values['Range'][0]) + values['Range'][0])
            elif (values['Type'] == "integer"):
                temp = np.linspace(0, 1, len(values['Mapping']) + 1)
                for j in range(1, len(temp)):
                    para_set.loc[(para_set_ud[item + "_UD"] >= (temp[j - 1] - EPS))
                                 & (para_set_ud[item + "_UD"] < (temp[j] + EPS)), item] = values['Mapping'][j - 1]
                para_set.loc[np.abs(para_set_ud[item + "_UD"] - 1) <= EPS, item] = values['Mapping'][-1]
                para_set[item] = para_set[item].round().astype(int)
            elif (values['Type'] == "categorical"):
                column_bool = [
                    item == para_name[::-1].split("DU_", maxsplit=1)[1][::-1] for para_name in self.para_ud_names]
                col_index = np.argmax(
                    para_set_ud.loc[:, column_bool].values, axis=1).tolist()
                para_set[item] = np.array(values['Mapping'])[col_index]
        return MappingData(para_set, logs_append=None)

    def _generate_init_design(self):
        """
        This function generates the initial uniform design.

        Returns
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point,
                and columns are used to represent variables.
        """

        self.logs = pd.DataFrame()
        ud_space = np.repeat(np.linspace(1 / (2 * self.n_runs_per_stage), 1 - 1 / (2 * self.n_runs_per_stage),
                              self.n_runs_per_stage).reshape([-1, 1]),
                             self.extend_factor_number, axis=1)

        base_ud = pydoe.design_query(n=self.n_runs_per_stage, s=self.extend_factor_number,
                                     q=self.n_runs_per_stage, crit="CD2", show_crit=False)
        if base_ud is None:
            base_ud = pydoe.gen_ud_ms(n=self.n_runs_per_stage, s=self.extend_factor_number, q=self.n_runs_per_stage, crit="CD2",
                              maxiter=self.max_search_iter, random_state=self.random_state, n_jobs=10, nshoot=10)

        if (not isinstance(base_ud, np.ndarray)):
            raise ValueError('Uniform design is not correctly constructed!')

        para_set_ud = np.zeros((self.n_runs_per_stage, self.extend_factor_number))
        for i in range(self.factor_number):
            loc_min = np.sum(self.variable_number[:(i + 1)])
            loc_max = np.sum(self.variable_number[:(i + 2)])
            for k in range(int(loc_min), int(loc_max)):
                para_set_ud[:, k] = ud_space[base_ud[:, k] - 1, k]
        para_set_ud = pd.DataFrame(para_set_ud, columns=self.para_ud_names)
        return para_set_ud

    def _generate_augment_design(self, ud_center):
        """
        This function refines the search space to a subspace of interest, and
        generates augmented uniform designs given existing designs.


        Parameters
        ----------
        ud_center: A numpy vector representing the center of the subspace,
               and corresponding elements denote the position of the center for each variable.

        Returns
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point,
                and columns are used to represent variables.
        """

        # 1. Transform the existing Parameters to Standardized Horizon (0-1)
        ud_space = np.zeros((self.n_runs_per_stage, self.extend_factor_number))
        ud_grid_size = 1.0 / (self.n_runs_per_stage * 2**(self.stage - 1))
        left_radius = np.floor((self.n_runs_per_stage - 1) / 2) * ud_grid_size
        right_radius = (self.n_runs_per_stage - np.floor((self.n_runs_per_stage - 1) / 2) - 1) * ud_grid_size
        for i in range(self.extend_factor_number):
            if ((ud_center[i] - left_radius) < (0 - EPS)):
                lb = 0
                ub = ud_center[i] + right_radius - (ud_center[i] - left_radius)
            elif ((ud_center[i] + right_radius) > (1 + EPS)):
                ub = 1
                lb = ud_center[i] - left_radius - \
                    (ud_center[i] + right_radius - 1)
            else:
                lb = max(ud_center[i] - left_radius, 0)
                ub = min(ud_center[i] + right_radius, 1)
            ud_space[:, i] = np.linspace(lb, ub, self.n_runs_per_stage)

        # 2. Map existing Runs' Parameters to UD Levels "x0" (1 - n_runs_per_stage)
        flag = True
        for i in range(self.extend_factor_number):
            flag = flag & (
                self.logs.loc[:, self.para_ud_names].iloc[:, i] >= (ud_space[0, i] - EPS))
            flag = flag & (
                self.logs.loc[:, self.para_ud_names].iloc[:, i] <= (ud_space[-1, i] + EPS))
        x0 = self.logs.loc[flag, self.para_ud_names].values

        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                x0[i, j] = (
                    np.where(abs(x0[i, j] - ud_space[:, j]) <= EPS)[0][0] + 1)

        x0 = np.round(x0).astype(int)
        # 3. Delete existing UD points on the same levels grids
        for i in range(self.extend_factor_number):
            keep_list = []
            unique = np.unique(x0[:, i])
            for j in range(len(unique)):
                xx_loc = np.where(np.abs(x0[:, i] - unique[j]) <= EPS)[0].tolist()
                keep_list.extend(np.random.choice(xx_loc, 1))
            x0 = x0[keep_list, :].reshape([-1, self.extend_factor_number])

        # Return if the maximum run has been reached.
        if ((self.logs.shape[0] + self.n_runs_per_stage - x0.shape[0]) > self.max_runs):
            self.stop_flag = True
            if self.verbose > 0:
                print("Maximum number of runs reached, stop!")
            return

        if (x0.shape[0] >= self.n_runs_per_stage):
            self.stop_flag = True
            if self.verbose > 0:
                print("Search space already full, stop!")
            return

        if self.include_cv_folds and (self.get_n_trials() >= self.max_runs):
            self.stop_flag = True
            if self.verbose > 0:
                print("Maximum number of trials reached")

        # 4. Generate Sequential UD
        base_ud = pydoe.gen_aud_ms(x0, n=self.n_runs_per_stage, s=self.extend_factor_number, q=self.n_runs_per_stage, crit="CD2",
                                   maxiter=self.max_search_iter, random_state=self.random_state, n_jobs=10, nshoot=10)
        if (not isinstance(base_ud, np.ndarray)):
            raise ValueError('Uniform design is not correctly constructed!')

        base_ud_aug = base_ud[(x0.shape[0]):base_ud.shape[0],
                              :].reshape([-1, self.extend_factor_number])

        para_set_ud = np.zeros(
            (base_ud_aug.shape[0], self.extend_factor_number))
        for i in range(self.factor_number):
            loc_min = np.sum(self.variable_number[:(i + 1)])
            loc_max = np.sum(self.variable_number[:(i + 2)])
            for k in range(int(loc_min), int(loc_max)):
                para_set_ud[:, k] = ud_space[base_ud_aug[:, k] - 1, k]
        para_set_ud = pd.DataFrame(para_set_ud, columns=self.para_ud_names)
        return para_set_ud

    def _evaluate_runs(self, obj_func, para_set_ud):
        """
        This function evaluates the performance scores of given trials.


        Parameters
        ----------
        obj_func: A callable function. It takes the values stored in each trial as input parameters, and
               output the corresponding scores.
        para_set_ud: A pandas dataframe where each row represents a UD trial point,
                and columns are used to represent variables.
        """
        mapping_data = self._para_mapping(para_set_ud)
        para_set = mapping_data.para_set

        para_set_ud.columns = self.para_ud_names
        candidate_params = [{para_set.columns[j]: para_set.iloc[i, j]
                             for j in range(para_set.shape[1])}
                            for i in range(para_set.shape[0])]
        
        n_folds = self.get_n_cv_folds()
        n_trials = self.get_n_trials()
        post_n_trials = n_trials + len(candidate_params) * n_folds

        if self.include_cv_folds and (post_n_trials > self.max_runs):
            delta = self.max_runs - n_trials
            batch = math.ceil(delta / n_folds)
            print(f"Evaluating all candidate parameters ({len(candidate_params)}) exedes the maximum number of trials ({self.max_runs}). Selecting the first {batch} candidates!")
            candidate_params = candidate_params[:batch + 1]

        if self.n_jobs > 1:
            out = Parallel(n_jobs=self.n_jobs)(
                    delayed(obj_func)(self.stage, i, len(candidate_params), n_folds, parameters) 
                    for i, parameters in enumerate(candidate_params)
                )
        else:
            out = []
            for i, parameters in enumerate(candidate_params):
                out.append(obj_func(self.stage, i, len(candidate_params), n_folds, parameters))
            out = np.array(out)

        logs_aug = para_set_ud.to_dict()
        
        if mapping_data.logs_append is not None:
            logs_aug.update(mapping_data.logs_append)

        logs_aug.update(para_set)
        logs_aug.update(pd.DataFrame(out, columns=["score"]))
        logs_aug = pd.DataFrame(logs_aug)
        logs_aug["stage"] = self.stage
        self.logs = pd.concat([self.logs, logs_aug]).reset_index(drop=True)
        if self.verbose > 0:
            print("Stage %d completed (%d/%d) with best score: %.5f."
                  % (self.stage, self.logs.shape[0], self.max_runs, self.logs["score"].max()))

    def _run(self, obj_func):
        """
        This function controls the procedures for implementing the sequential uniform design method.

        Parameters
        ----------
        obj_func: A callable function. It takes the values stored in each trial as input parameters, and
               output the corresponding scores.
        """

        para_set_ud = self._generate_init_design()
        self._evaluate_runs(obj_func, para_set_ud)
        self.stage += 1
        while (True):
            ud_center = self.logs.sort_values(
                "score", ascending=False).loc[:, self.para_ud_names].values[0, :]
            para_set_ud = self._generate_augment_design(ud_center)
            if not self.stop_flag:
                self._evaluate_runs(obj_func, para_set_ud)
                self.stage += 1
            else:
                break

    def fmax(self, wrapper_func):
        """
        Search the optimal value of a function.

        Parameters
        ----------
        :type func: callable function
        :param func: the function to be optimized.

        """

        self.stage = 1
        np.random.seed(self.random_state)
        search_start_time = time.time()
        self._run(wrapper_func)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time
        self._summary()

    def fit(self, x, y=None, fit_params: dict = None):

        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        :type x: array, shape = [n_samples, n_features]
        :param x: input variales.

        :type y: array, shape = [n_samples] or [n_samples, n_output], optional
        :param y: target variable.

        """

        def sklearn_wrapper(stage, i, runs, n_folds, parameters):
            self.estimator.set_params(**parameters)
            start = time.perf_counter()
            out = cross_val_score(self.estimator, x, y,
                           cv=self.cv, scoring=self.scoring, fit_params=fit_params)
            score = np.mean(out)
            end = time.perf_counter() - start

            if self.verbose == 2:
                print(
                    f"Stage {stage}: runs=({i}/{runs}), trials={(i + 1) * n_folds} score={round(score, 6)}, " 
                    f"time={round(end, 2)}->{time_to_str(end)}, params={parameters}"
                )
            return score

        self.stage = 1
        self.logs = pd.DataFrame()
        np.random.seed(self.random_state)
        index = np.where(["random_state" in param for param in list(self.estimator.get_params().keys())])[0]
        for idx in index:
            self.estimator.set_params(**{list(self.estimator.get_params().keys())[idx]:self.random_state})

        search_start_time = time.time()
        self._run(sklearn_wrapper)
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time
        self._summary()

        if self.refit:
            self.best_estimator_ = self.estimator.set_params(
                **self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(x, y)
            else:
                self.best_estimator_.fit(x)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time