import pandas as pd
import numpy as np
from typing import List
import ray
import os
from collections import ChainMap


class InfluenceErrorSignals:

    def __init__(self):
        self.__self_influence = {
            'SI': self.si,
        }
        self.__marginal_signals = {
            'MPI': self.mpi,
            'MNI': self.mni,
            'MAI': self.mai,
        }
        self.__conditional_signals = {
            'PIL': self.pil,
            'NIL': self.nil,
            'AIL': self.ail,
            '#SLIP': self.num_slip,
            '#SLIN': self.num_slin
        }

    def compute_signals_per_sample(self, inf_mat, train_samples_to_examine, y_true, signals, njobs=1):
        if njobs > 1:
            njobs = max(njobs, os.cpu_count())
            if ray.is_initialized():
                ray.shutdown()
            else:
                ray.init()
            samples_splitted = np.array_split(train_samples_to_examine, njobs)
            results = ray.get(
                [InfluenceErrorSignals._compute_signals_per_sample_distributed.
                     remote(inf_mat, sample_chunk, y_true, signals) for sample_chunk in samples_splitted])
            signals_per_sample = dict(ChainMap(*results))
            signals_df = pd.DataFrame.from_dict(signals_per_sample, orient='index').sort_index()
        else:
            signals_per_sample = {}
            print('Error signals computation for {} samples'.format(len(train_samples_to_examine)))
            for sample_id in train_samples_to_examine:
                inf_vec = inf_mat.loc[sample_id]
                signals_per_sample[sample_id] = self.compute_signals(signal_names=signals, inf_vec_of_x=inf_vec,
                                                                     x_id=sample_id,
                                                                     y_vector=y_true.loc[inf_mat.columns],
                                                                     y_xid=y_true.loc[sample_id])
            signals_df = pd.DataFrame.from_dict(signals_per_sample, orient='index').sort_index()
        return signals_df

    @staticmethod
    @ray.remote
    def _compute_signals_per_sample_distributed(inf_mat, samples_to_examine, y_true, signals):
        ies = InfluenceErrorSignals()
        signals_per_sample = {}
        print('Error signals computation for {} samples'.format(len(samples_to_examine)))
        for sample_id in samples_to_examine:
            inf_vec = inf_mat.loc[sample_id]
            signals_per_sample[sample_id] = ies.compute_signals(signal_names=signals, inf_vec_of_x=inf_vec,
                                                                x_id=sample_id,
                                                                y_vector=y_true.loc[inf_mat.columns],
                                                                y_xid=y_true.loc[sample_id])
        return signals_per_sample

    def compute_all_signals(self, inf_vec_of_x: pd.Series, x_id: int, y_vector: pd.Series, y_xid: int):
        signal_vals = {}
        signal_vals['SI'] = self.__self_influence['SI'](inf_vec_of_x, x_id)
        for sig, fn in self.__marginal_signals.items():
            signal_vals[sig] = fn(inf_vec_of_x, y_vector)
        for sig, fn in self.__conditional_signals.items():
            signal_vals[sig] = fn(inf_vec_of_x, y_vector, y_xid)
        return signal_vals

    def compute_signals(self, signal_names: List[str], inf_vec_of_x: pd.Series, x_id: int, y_vector: pd.Series, y_xid: int):
        signal_vals = {}
        for sig_name in signal_names:
            if sig_name in self.__self_influence:
                signal_vals[sig_name] = self.__self_influence[sig_name](inf_vec_of_x, x_id)
            elif sig_name in self.__marginal_signals:
                signal_vals[sig_name] = self.__marginal_signals[sig_name](inf_vec_of_x, y_vector)
            elif sig_name in self.__conditional_signals:
                signal_vals[sig_name] = self.__conditional_signals[sig_name](inf_vec_of_x, y_vector, y_xid)
            else:
                raise ValueError('{} not in known signals {}'.format(sig_name, self.signals_names()))
        return signal_vals

    def signals_names(self):
        return {
            *self.__self_influence.keys(),
            *self.__marginal_signals.keys(),
            *self.__conditional_signals.keys()
        }

    def si(self, inf_vec_of_x: pd.Series, x_id: int):
        return inf_vec_of_x.loc[x_id]

    def pil(self, inf_vec_of_x: pd.Series, y_vector: pd.Series, y: int):
        assert len(inf_vec_of_x) == len(y_vector)
        inf_vec_of_x = inf_vec_of_x.loc[self.__get_samples_of_class(y_vector=y_vector, y=y)]
        return inf_vec_of_x[inf_vec_of_x > 0].sum()

    def nil(self, inf_vec_of_x: pd.Series, y_vector: pd.Series, y: int):
        assert len(inf_vec_of_x) == len(y_vector)
        inf_vec_of_x = inf_vec_of_x.loc[self.__get_samples_of_class(y_vector=y_vector, y=y)]
        return - inf_vec_of_x[inf_vec_of_x < 0].sum()

    def ail(self, inf_vec_of_x: pd.Series, y_vector: pd.Series, y: int):
        assert len(inf_vec_of_x) == len(y_vector)
        return self.pil(inf_vec_of_x=inf_vec_of_x, y_vector=y_vector, y=y) + \
               self.nil(inf_vec_of_x=inf_vec_of_x, y_vector=y_vector, y=y)

    def mpi(self, inf_vec_of_x: pd.Series, y_vector: pd.Series):
        assert len(inf_vec_of_x) == len(y_vector)
        marginal_pil = 0
        for y in np.unique(y_vector):
            marginal_pil += self.pil(inf_vec_of_x=inf_vec_of_x, y_vector=y_vector, y=y)
        return marginal_pil

    def mni(self, inf_vec_of_x: pd.Series, y_vector: pd.Series):
        assert len(inf_vec_of_x) == len(y_vector)
        marginal_nil = 0
        for y in np.unique(y_vector):
            marginal_nil += self.nil(inf_vec_of_x=inf_vec_of_x, y_vector=y_vector, y=y)
        return marginal_nil

    def mai(self, inf_vec_of_x: pd.Series, y_vector: pd.Series):
        assert len(inf_vec_of_x) == len(y_vector)
        marginal_ail = 0
        for y in np.unique(y_vector):
            marginal_ail += self.ail(inf_vec_of_x=inf_vec_of_x, y_vector=y_vector, y=y)
        return marginal_ail

    def num_slip(self, inf_vec_of_x: pd.Series, y_vector: pd.Series, y: int):
        assert len(inf_vec_of_x) == len(y_vector)
        inf_vec_of_x = inf_vec_of_x.loc[self.__get_samples_of_class(y_vector=y_vector, y=y)]
        return len(inf_vec_of_x[inf_vec_of_x > 0].index)

    def num_slin(self, inf_vec_of_x: pd.Series, y_vector: pd.Series, y: int):
        assert len(inf_vec_of_x) == len(y_vector)
        inf_vec_of_x = inf_vec_of_x.loc[self.__get_samples_of_class(y_vector=y_vector, y=y)]
        return len(inf_vec_of_x[inf_vec_of_x < 0].index)

    def __get_samples_of_class(self, y_vector, y):
        return y_vector[y_vector == y].index
