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
import os
import joblib
import copy
from typing import Union, List

import inspyred
import datetime

import joblib
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
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import one_hot

import torch_explain as te
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, complexity


class EvoLENs(BaseEstimator, TransformerMixin):
    """
    EvoFS class.
    """

    def __init__(self, lens: torch.nn.Module, optimizer_name, loss_form, lr: float = 0.01,
                 compression: str = 'both', pretrain: bool = True,
                 pop_size: int = 100, max_generations: int = 100,
                 max_features: int = 100, min_features: int = 10,
                 max_samples: int = 500, min_samples: int = 50,
                 n_splits: int = 3, random_state: int = 42,
                 train_epochs: int = 10, reset_generations: int = 2,
                 scoring: str = 'f1_weighted', trainval_index: torch.Tensor = None,
                 test_index: torch.Tensor = None, verbose: bool = True):

        self.lens = lens
        self.optimizer_name = optimizer_name
        self.loss_form = loss_form
        self.lr = lr
        self.compression = compression
        self.pretrain = pretrain
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
        self.trainval_index = trainval_index
        self.test_index = test_index
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
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
        self.reset_generations_ = 0
        self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)
        self.train_epochs_ = self.train_epochs

        self.x_, self.y_ = torch.FloatTensor(X), torch.LongTensor(y)
        self.y1h_ = one_hot(torch.LongTensor(self.y_)).float()
        if self.trainval_index is None or self.test_index is None:
            self.skf_trainval_test_ = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
            self.trainval_index_, self.test_index_ = next(self.skf_trainval_test_.split(self.x_, self.y_))
        else:
            self.trainval_index_, self.test_index_ = self.trainval_index, self.test_index

        self.skf_train_val_ = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        self.train_index_, self.val_index_ = next(self.skf_train_val_.split(self.x_[self.trainval_index_], self.y_[self.trainval_index_]))

        if self.pretrain:
            self.train_epochs_ = 4000
            # # for layer in self.lens.children():
            # #     if hasattr(layer, 'reset_parameters'):
            # #         layer.reset_parameters()
            # self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)
            candidate = np.arange(0, self.x_.shape[1])
            self._evaluate_candidate(candidate, train=True)
            f1, explanation_accuracy, explanation_complexity, explanations = self._evaluate_candidate(candidate, train=False)
            # for layer in self.lens.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
            # self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)
            self.train_epochs_ = self.train_epochs

            print('Initial F1: {:.2f}'.format(f1))
            print('Initial explanation accuracy: {:.2f}'.format(explanation_accuracy))
            print('Initial explanation complexity: {:.2f}'.format(explanation_complexity))
            print('Initial explanations: {}'.format(explanations))

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.random_state)

        self.ea = inspyred.ec.emo.NSGA2(prng)
        self.ea.variator = [self._variate]
        self.ea.terminator = inspyred.ec.terminators.generation_termination
        self.ea.observer = self._observe

        self.ea.evolve(
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
        f1_best = 0
        self.optimal_solutions_ = []
        feature_counts = np.zeros(X.shape[1])
        self.train_epochs_ = 4000
        for individual in self.ea.archive:
            candidate = individual.candidate[1]
            feature_counts[candidate] += 1

            for layer in self.lens.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)
            self._evaluate_candidate(candidate, train=True)
            f1, explanation_accuracy, explanation_complexity, explanations = self._evaluate_candidate(candidate,
                                                                                                      train=False)

            if f1_best < f1:
                self.best_set_ = {
                    'features': candidate,
                    'accuracy': f1,
                    'explanation_accuracy': explanation_accuracy,
                    'explanation_complexity': explanation_complexity,
                    'explanations': explanations,
                }
                f1_best = f1

            self.optimal_solutions_.append({
                'features': candidate,
                'accuracy': f1,
                'explanation_accuracy': explanation_accuracy,
                'explanation_complexity': explanation_complexity,
                'explanations': explanations,
            })

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
            individual_f = np.random.choice(self.x_.shape[1], size=(n_features,), replace=False).tolist()
            individual_f = np.sort(individual_f).tolist()

        if self.compression == 'samples' or self.compression == 'both':
            n_samples = random.randint(self.min_samples, self.max_samples_)
            individual_s = np.random.choice(self.x_.shape[0], size=(n_samples,), replace=False).tolist()
            individual_s = np.sort(individual_s).tolist()

        individual = [individual_s, individual_f]

        return individual

    # using inspyred's notation, here is a single operator that performs both crossover and mutation, sequentially
    def _variate(self, random, candidates, args):
        nextgen_f, nextgen_s = [[] for _ in range(len(candidates))], [[] for _ in range(len(candidates))]
        if self.compression == 'features' or self.compression == 'both':
            candidates_f = [c[1] for c in candidates]
            nextgen_f = self._do_variation(random, candidates_f, self.min_features,
                                           self.max_features_, self.x_.shape[1], args)

        if self.compression == 'samples' or self.compression == 'both':
            candidates_s = [c[0] for c in candidates]
            nextgen_s = self._do_variation(random, candidates_s, self.min_samples,
                                           self.max_samples_, self.x_.shape[0], args)

        next_generation = [[cs, cf] for cs, cf in zip(nextgen_s, nextgen_f)]
        return next_generation

    def _do_variation(self, random, candidates, min_candidate_size, max_candidate_size, max_size, args):
        split_idx = int(len(candidates) / 2)
        fathers = candidates[:split_idx]
        mothers = candidates[split_idx:]
        next_generation = []
        parent = np.zeros((max_size), dtype=int)

        for father, mother in zip(fathers, mothers):
            parent1 = 0 * parent
            parent1[father] = 1
            parent2 = 0 * parent
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
                points_selected = list(np.argwhere(child == 1).squeeze(-1))
                points_not_selected = list(np.argwhere(child == 0).squeeze(-1))

                if len(points_selected) > max_candidate_size:
                    index = np.random.choice(points_selected, len(points_selected) - max_candidate_size)
                    child[index] = 0

                if len(points_selected) < min_candidate_size:
                    index = np.random.choice(points_not_selected, min_candidate_size - len(points_selected))
                    child[index] = 1

                points_selected = list(np.argwhere(child == 1).squeeze(-1))
                next_gen.append(points_selected)

            next_generation.append(next_gen[0])
            next_generation.append(next_gen[1])

        return next_generation

    def _evaluate_candidate(self, candidate, train):
        x_reduced = torch.zeros_like(self.x_)
        x_reduced[:, candidate] = self.x_[:, candidate]

        train_index = np.random.choice(self.trainval_index_, size=int(0.8*len(self.trainval_index_)), replace=False)
        val_mask = torch.ones(len(self.x_), dtype=torch.bool)
        val_mask[train_index] = False
        val_mask[self.test_index_] = False
        val_index = torch.where(val_mask)[0]

        # train loop
        if train and self.pretrain:
            self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)
            self.lens.train()
            for epoch in range(self.train_epochs_):
                self.optimizer_.zero_grad()
                # y_pred = self.lens(x_reduced).squeeze(-1)
                y_pred = self.lens(self.x_).squeeze(-1)
                # loss = loss_form(y_pred[train_index], self.y_[train_index])
                loss = self.loss_form(y_pred, self.y_)
                loss.backward()
                self.optimizer_.step()
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{self.train_epochs_}: {loss.item()}')
        else:
            self.lens.eval()
            y_pred = self.lens(x_reduced).squeeze(-1)

        if train:
            f1 = f1_score(self.y_[val_mask], y_pred[val_mask].argmax(axis=-1), average='weighted')
            explanations = entropy.explain_classes(self.lens, self.x_, self.y1h_, train_index,
                                                   val_mask=train_index,
                                                   test_mask=val_index,
                                                   c_threshold=0., y_threshold=0,
                                                   topk_explanations=100, max_minterm_complexity=5)
            explanation_accuracy = np.min([exp_dict['explanation_accuracy'] for exp_dict in explanations.values()])
            explanation_complexity = np.max([exp_dict['explanation_complexity'] for exp_dict in explanations.values()])
        else:
            f1 = f1_score(self.y_[self.test_index_], y_pred[self.test_index_].argmax(axis=-1), average='weighted')
            explanations = entropy.explain_classes(self.lens, self.x_, self.y1h_, self.trainval_index_,
                                                   val_mask=self.trainval_index_, test_mask=self.test_index_,
                                                   c_threshold=0., y_threshold=0,
                                                   topk_explanations=100, max_minterm_complexity=5)
            explanation_accuracy = np.median([exp_dict['explanation_accuracy'] for exp_dict in explanations.values()])
            explanation_complexity = np.median([exp_dict['explanation_complexity'] for exp_dict in explanations.values()])

        return f1, explanation_accuracy, explanation_complexity, explanations

    # function that evaluates the feature sets
    def _evaluate(self, candidates, args):
        fitness = []
        for cid, c in enumerate(candidates):
            f1, explanation_accuracy, explanation_complexity, _ = self._evaluate_candidate(c[1], train=True)
            objectives = [1 - f1, 1 - explanation_accuracy, explanation_complexity]
            fitness.append(inspyred.ec.emo.Pareto(objectives))

            if self.verbose:
                print(f'\t Candidate {cid}/{len(candidates)}: {objectives}')

        if self.reset_generations_ > self.reset_generations:
            for layer in self.lens.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            self.optimizer_ = torch.optim.AdamW(self.lens.parameters(), lr=self.lr)
            self.reset_generations_ = 0
        else:
            self.reset_generations_ += 1

        return fitness

    # the 'observer' function is called by inspyred algorithms at the end of every generation
    def _observe(self, population, num_generations, num_evaluations, args):
        old_time = args["current_time"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        best_candidate_id = np.argmin(np.array([candidate.fitness[0] for candidate in args['_ec'].archive]))
        best_candidate = args['_ec'].archive[best_candidate_id]
        log = f"\n[{delta_time_string}] Generation {num_generations}/{self.max_generations_}, " \
              f"Best individual (model): " \
              f"error={best_candidate.fitness[0] * 100:.2f}, " \
              f"exp_error={best_candidate.fitness[1] * 100:.2f}, " \
              f"exp_complexity={best_candidate.fitness[2]:.2f}\n\n" \
              f"New generation...\n"

        if self.verbose:
            print(log)

        best_candidate_id = np.argmin(np.array([candidate.fitness[1] for candidate in args['_ec'].archive]))
        best_candidate = args['_ec'].archive[best_candidate_id]
        log = f"Best individual (explanation): " \
              f"error={best_candidate.fitness[0] * 100:.2f}, " \
              f"exp_error={best_candidate.fitness[1] * 100:.2f}, " \
              f"exp_complexity={best_candidate.fitness[2]:.2f}\n\n" \
              f"New generation...\n"

        if self.verbose:
            print(log)

        args["current_time"] = current_time


if __name__ == '__main__':
    x, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_classes=2,
                               random_state=42, )
    # x, y = load_iris(return_X_y=True)
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    layers = [
        te.nn.EntropyLinear(x.shape[1], 100, n_classes=len(np.unique(y))),
        # torch.nn.Linear(x.shape[1], 100),
        torch.nn.LeakyReLU(),
        # torch.nn.Linear(30, 10),
        # torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
        # torch.nn.Linear(100, len(np.unique(y))),
    ]
    model = torch.nn.Sequential(*layers)
    optimizer = 'adamw'  #
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    # loss_form = torch.nn.BCEWithLogitsLoss()
    loss_form = torch.nn.CrossEntropyLoss()
    # for layer in model.children():
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    # model.train()
    # for epoch in range(10000):
    #     optim.zero_grad()
    #     y_pred = model(torch.FloatTensor(x)).squeeze(-1)
    #     loss = loss_form(y_pred, torch.LongTensor(y))
    #     loss.backward()
    #     optim.step()
    #     if epoch % 100 == 0:
    #         print(f'Epoch {epoch}/{10000}: {loss.item()}')

    evo_lens = EvoLENs(model, optimizer, loss_form, compression='features', train_epochs=10,
                       max_generations=100, lr=0.001,)
    evo_lens.fit(x, y)

    result_dir = './evolens_results/'
    os.makedirs(result_dir, exist_ok=True)
    joblib.dump(evo_lens, f'{result_dir}evo_lens.pkl')
    evo_lens2 = joblib.load(f'{result_dir}evo_lens.pkl')
    joblib.dump(evo_lens.best_set_, f'{result_dir}evo_lens_best.joblib')
    joblib.dump(evo_lens.optimal_solutions_, f'{result_dir}evo_lens_solutions.joblib')
    joblib.dump(evo_lens.feature_ranking_, f'{result_dir}evo_lens_feature_ranking.joblib')

    print()
    print('Best individual\'s features:')
    print(evo_lens.best_set_['features'])
    print()
    print('Best individual\'s F1:')
    print(evo_lens.best_set_['accuracy'])
    print()
    print('Best individual\'s explanation accuracy:')
    print(evo_lens.best_set_['explanation_accuracy'])
    print()
    print('Best individual\'s explanation complexity:')
    print(evo_lens.best_set_['explanation_complexity'])
    print()
