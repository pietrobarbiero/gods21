# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero, Giovanni Squillero and Alberto Tonda
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import copy
from typing import Union, List

import inspyred
import datetime
import numpy as np
import multiprocessing

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import get_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
import warnings
import pandas as pd

import torch
from torch.nn.functional import one_hot

import torch_explain as te
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, complexity


class EvoLENs(BaseEstimator, TransformerMixin):
    """
    EvoFS class.
    """

    def __init__(self, lens: torch.nn.Module, optimizer_name, loss_form, lr: float = 0.001,
                 compression: str = 'both',
                 pop_size: int = 100, max_generations: int = 100,
                 max_features: int = 100, min_features: int = 10,
                 max_samples: int = 500, min_samples: int = 50,
                 n_splits: int = 3, random_state: int = 42,
                 train_epochs: int = 10, reset_generations: int = 50,
                 scoring: str = 'f1_weighted', verbose: bool = True):

        self.lens = lens
        self.optimizer_name = optimizer_name
        self.loss_form = loss_form
        self.lr = lr
        self.compression = compression
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_features = max_features
        self.min_features = min_features
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring
        self.train_epochs = train_epochs
        self.reset_generations = reset_generations
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n = int(X.shape[0])
        k = int(X.shape[1])

        self.max_generations_ = np.min([self.max_generations, int(math.log10(2 ** int(0.01 * k * n)))])
        self.pop_size_ = np.min([self.pop_size, int(math.log10(2 ** k))])
        self.offspring_size_ = 2 * self.pop_size_
        self.maximize_ = False
        self.individuals_ = []
        self.scorer_ = get_scorer(self.scoring)
        self.max_features_ = np.min([k, self.max_features])
        self.min_features_ = np.min([self.min_features, self.max_features_])
        self.max_samples_ = np.min([n, self.max_samples])
        self.reset_generations_ = np.min([self.max_generations_, self.reset_generations])
        self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        list_of_splits = [split for split in skf.split(X, y)]
        trainval_index, test_index = list_of_splits[0]
        self.x_trainval_, x_test = X.iloc[trainval_index], X.iloc[test_index]
        self.y_trainval_, y_test = y[trainval_index], y[test_index]

        self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        # list_of_splits2 = [split for split in self.skf.split(self.x_trainval_, self.y_trainval_)]
        # train_index, val_index = list_of_splits2[0]
        # self.x_train_, self.x_val = self.x_trainval_.iloc[train_index], self.x_trainval_.iloc[val_index]
        # self.y_train_, self.y_val = self.y_trainval_[train_index], self.y_trainval_[val_index]

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.random_state)

        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [self._variate]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = self._observe

        ea.evolve(
            generator=self._generate,

            evaluator=self._evaluate,
            # this part is defined to use multi-process evaluations
            # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
            # mp_evaluator=self._evaluate_feature_sets,
            # mp_num_cpus=multiprocessing.cpu_count()-2,

            pop_size=self.pop_size_,
            num_selected=self.offspring_size_,
            maximize=self.maximize_,
            max_generations=self.max_generations_,

            # extra arguments here
            current_time=datetime.datetime.now()
        )

        print('Training completed!')

        # find best individual, the one with the highest accuracy on the validation set
        accuracy_best = 0
        self.solutions_ = []
        feature_counts = np.zeros(X.shape[1])
        for individual in ea.archive:

            feature_set = individual.candidate[1]
            feature_counts[feature_set] += 1

            if self.compression == 'features':
                x_reduced = self.x_trainval_.iloc[:, individual.candidate[1]]
                y_reduced = self.y_trainval_
                x_test_reduced = x_test.iloc[:, individual.candidate[1]]
            elif self.compression == 'samples':
                x_reduced = self.x_trainval_.iloc[individual.candidate[0]]
                y_reduced = self.y_trainval_[individual.candidate[0]]
                x_test_reduced = x_test
            elif self.compression == 'both':
                x_reduced = self.x_trainval_.iloc[individual.candidate[0], individual.candidate[1]]
                y_reduced = self.y_trainval_[individual.candidate[0]]
                x_test_reduced = x_test.iloc[:, individual.candidate[1]]

            model = copy.deepcopy(self.estimator)
            model.fit(x_reduced, y_reduced)

            # compute validation accuracy
            accuracy_test = self.scorer_(model, x_test_reduced, y_test)

            if accuracy_best < accuracy_test:
                self.best_set_ = {
                    'samples': individual.candidate[0],
                    'features': individual.candidate[1],
                    'accuracy': accuracy_test,
                }
                accuracy_best = accuracy_test

            individual.validation_score_ = accuracy_test
            self.solutions_.append(individual)

        self.feature_ranking_ = np.argsort(feature_counts)
        return self

    def transform(self, X, **fit_params):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if self.compression == 'features':
            return X.iloc[:, self.best_set_['features']].values
        elif self.compression == 'samples':
            return X.iloc[self.best_set_['samples']].values
        elif self.compression == 'both':
            return X.iloc[self.best_set_['samples'], self.best_set_['features']].values

    # initial random generation of feature sets
    def _generate(self, random, args):
        individual_f, individual_s = [], []

        if self.compression == 'features' or self.compression == 'both':
            n_features = random.randint(self.min_features_, self.max_features_)
            individual_f = np.random.choice(self.x_trainval_.shape[1], size=(n_features,), replace=False).tolist()
            individual_f = np.sort(individual_f).tolist()

        if self.compression == 'samples' or self.compression == 'both':
            n_samples = random.randint(self.min_samples, self.max_samples_)
            individual_s = np.random.choice(self.x_trainval_.shape[0], size=(n_samples,), replace=False).tolist()
            individual_s = np.sort(individual_s).tolist()

        individual = [individual_s, individual_f]

        return individual

    # using inspyred's notation, here is a single operator that performs both crossover and mutation, sequentially
    def _variate(self, random, candidates, args):
        nextgen_f, nextgen_s = [[] for _ in range(len(candidates))], [[] for _ in range(len(candidates))]
        if self.compression == 'features' or self.compression == 'both':
            candidates_f = [c[1] for c in candidates]
            nextgen_f = self._do_variation(random, candidates_f, self.min_features,
                                           self.max_features_, self.x_trainval_.shape[1], args)

        if self.compression == 'samples' or self.compression == 'both':
            candidates_s = [c[0] for c in candidates]
            nextgen_s = self._do_variation(random, candidates_s, self.min_samples,
                                           self.max_samples_, self.x_trainval_.shape[0], args)

        next_generation = [[cs, cf] for cs, cf in zip(nextgen_s, nextgen_f)]
        return next_generation

    def _do_variation(self, random, candidates, min_candidate_size, max_candidate_size, max_size, args):
        split_idx = int(len(candidates) / 2)
        fathers = candidates[:split_idx]
        mothers = candidates[split_idx:]
        next_generation = []
        parent = np.zeros((max_size), dtype=int)

        for father, mother in zip(fathers, mothers):
            parent1 = 0*parent
            parent1[father] = 1
            parent2 = 0*parent
            parent2[mother] = 1

            # well, for starters we just crossover two individuals, then mutate
            children = [list(parent1), list(parent2)]

            # one-point crossover!
            cut_point = random.randint(0, len(children[0]) - 1)
            for index in range(0, cut_point + 1):
                temp = children[0][index]
                children[0][index] = children[1][index]
                children[1][index] = temp

            # mutate!
            for child in children:
                mutation_point = random.randint(0, len(child) - 1)
                if child[mutation_point] == 0:
                    child[mutation_point] = 1
                else:
                    child[mutation_point] = 0

            # check if individual is still valid, and (in case it isn't) repair it
            next_gen = []
            for child in children:
                child = np.array(child)
                points_selected = list(np.argwhere(child == 1).squeeze())
                points_not_selected = list(np.argwhere(child == 0).squeeze())

                if len(points_selected) > max_candidate_size:
                    index = np.random.choice(points_selected, len(points_selected) - max_candidate_size)
                    child[index] = 0

                if len(points_selected) < min_candidate_size:
                    index = np.random.choice(points_not_selected, min_candidate_size - len(points_selected))
                    child[index] = 1

                points_selected = list(np.argwhere(child == 1).squeeze())
                next_gen.append(points_selected)

            next_generation.append(next_gen[0])
            next_generation.append(next_gen[1])

        return next_generation

    # function that evaluates the feature sets
    def _evaluate(self, candidates, args):
        fitness = []
        list_of_splits2 = [split for split in self.skf.split(self.x_trainval_, self.y_trainval_)]
        train_index, val_index = list_of_splits2[np.random.randint(0, self.skf.n_splits)]
        x_train_, x_val = self.x_trainval_.iloc[train_index], self.x_trainval_.iloc[val_index]
        y_train_, y_val = self.y_trainval_[train_index], self.y_trainval_[val_index]
        x_train_, x_val = torch.FloatTensor(x_train_.values), torch.FloatTensor(x_val.values)
        y_train_, y_val = one_hot(torch.LongTensor(y_train_)).float(), one_hot(torch.LongTensor(y_val)).float()
        for cid, c in enumerate(candidates):
            if self.compression == 'features':
                x_reduced = torch.zeros_like(x_train_)
                x_reduced[:, c[1]] = x_train_[:, c[1]]
                y_reduced = y_train_
                x_val_reduced = torch.zeros_like(x_val)
                x_val_reduced[:, c[1]] = x_val[:, c[1]]
            # elif self.compression == 'samples':
            #     x_reduced = x_train_[c[0]]
            #     y_reduced = y_train_[c[0]]
            #     x_val_reduced = x_val
            # elif self.compression == 'both':
            #     x_reduced = x_train_[c[0], c[1]]
            #     y_reduced = y_train_[c[0]]
            #     x_val_reduced = x_val[:, c[1]]

            # train loop
            self.lens.train()
            for epoch in range(self.train_epochs):
                self.optimizer_.zero_grad()
                y_pred = self.lens(x_reduced).squeeze(-1)
                loss = loss_form(y_pred, y_reduced)
                loss.backward()
                self.optimizer_.step()

            f1 = f1_score(y_reduced.argmax(dim=-1), y_pred.argmax(dim=-1), average='weighted')
            x = torch.FloatTensor(self.x_trainval_.values)
            y1h = one_hot(torch.LongTensor(self.y_trainval_)).float()
            explanations = entropy.explain_classes(self.lens, x, y1h, train_index, train_index,
                                                   c_threshold=0., y_threshold=0, topk_explanations=100,
                                                   max_minterm_complexity=5)
            explanation_accuracy = np.min([exp_dict['explanation_accuracy'] for exp_dict in explanations.values()])
            explanation_complexity = np.max([exp_dict['explanation_complexity'] for exp_dict in explanations.values()])

            # maximizing the points removed also means
            # minimizing the number of points taken (LOL)
            objectives = [1-f1, 1-explanation_accuracy, explanation_complexity]
            fitness.append(inspyred.ec.emo.Pareto(objectives))

            if self.verbose:
                print(f'Candidate {cid}/{len(candidates)}: {objectives}')

        return fitness

    # the 'observer' function is called by inspyred algorithms at the end of every generation
    def _observe(self, population, num_generations, num_evaluations, args):
        # sample_size = self.x_trainval_.shape[0]
        # feature_size = self.x_trainval_.shape[1]
        old_time = args["current_time"]
        # logger = args["logger"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        best_candidate_id = np.argmin(np.array([candidate.fitness[0] for candidate in args['_ec'].archive]))
        # best_candidate_id = np.argmax(np.array([candidate.fitness[2] for candidate in population]))
        best_candidate = args['_ec'].archive[best_candidate_id]
        # best_candidate = population[0]

        log = f"[{delta_time_string}] Generation {num_generations}, Best individual: "
        # if self.compression == 'samples' or self.compression == 'both':
        #     log += f"#samples={len(best_candidate.candidate[0])} (of {sample_size}), "
        # if self.compression == 'features' or self.compression == 'both':
        #     log += f"#features={len(best_candidate.candidate[1])} (of {feature_size}), "
        log += f"error={best_candidate.fitness[0]*100:.2f}, exp_error={best_candidate.fitness[1]*100:.2f}, exp_complexity={best_candidate.fitness[2]:.2f}"

        if self.verbose:
            print(log)
        #     logger.info(log)

        args["current_time"] = current_time


if __name__ == '__main__':
    x, y = make_classification(n_samples=500, n_features=30, n_informative=5, n_classes=4, random_state=42)

    layers = [
        te.nn.EntropyLinear(x.shape[1], 10, n_classes=len(np.unique(y))),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(10, 4),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(4, 1),
    ]
    model = torch.nn.Sequential(*layers)
    optimizer = 'adamw' #torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_form = torch.nn.BCEWithLogitsLoss()
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    evo_lens = EvoLENs(model, optimizer, loss_form, compression='features', train_epochs=10)
    evo_lens.fit(x, y)

    print()
