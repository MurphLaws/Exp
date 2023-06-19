from typing import Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from deel.influenciae.common import (ConjugateGradientDescentIHVP, ExactIHVP,
                                     InfluenceModel)
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from tensorflow.keras.losses import BinaryCrossentropy, Reduction


class FirstOrderInfluence:
    def __init__(
        self,
        model_obj: tf.keras.Model,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        random_state: int = None,
        unreduced_loss_fn: Any = BinaryCrossentropy(reduction=Reduction.NONE),
    ):
        # constructor arguments
        self.model_obj = model_obj
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.unreduced_loss_fn = unreduced_loss_fn

        # class variables
        self._influence_func_instance: Optional[FirstOrderInfluenceCalculator] = None
        self._inf_models_are_built = False

    def build_influence_models(
        self, train_data: tf.data.Dataset, val_data: tf.data.Dataset
    ):
        print("Building First Order Influence")

        self.model_obj.fit(
            train_data.batch(self.batch_size),
            epochs=self.epochs,
            validation_data=val_data.batch(self.batch_size),
        )
        influence_model = InfluenceModel(
            self.model_obj,
            start_layer=-1,
            loss_function=self.unreduced_loss_fn
            # always set the reduction to none when computing influences
        )
        # ihvp_calculator = ConjugateGradientDescentIHVP(model=influence_model, extractor_layer=-1, train_dataset=train_data.batch(self.batch_size))
        # self._influence_func_instance = FirstOrderInfluenceCalculator(
        #     model=influence_model, dataset=train_data, ihvp_calculator=ihvp_calculator
        # )

        # This is with exact hessian vector product, which is more Memory and computationaly intensive
        ihvp_calculator = ExactIHVP(influence_model, train_data.batch(16))
        self._influence_func_instance = FirstOrderInfluenceCalculator(
            model=influence_model, dataset=train_data, ihvp_calculator=ihvp_calculator
        )

        print("First Order Influence building complete.")
        self._inf_models_are_built = True

    def compute_self_influence(self, train_points: tf.data.Dataset) -> pd.DataFrame:
        if not self._inf_models_are_built:
            raise AssertionError(
                "Influence Models are not created. Call the function build_influence_models"
            )
        influences = self._influence_func_instance.compute_influence_values(
            train_set=train_points
        )
        return pd.DataFrame(np.diag(pd.DataFrame([e[1] for e in influences][0]).values.flatten()))

    def compute_train_to_test_influence(self, train_points: tf.data.Dataset,
                                        test_points: tf.data.Dataset) -> pd.DataFrame:
        if not self._inf_models_are_built:
            raise AssertionError(
                "Influence Models are not created. Call the function build_influence_models"
            )
        influences = (
            self._influence_func_instance.estimate_influence_values_in_batches(
                dataset_to_evaluate=test_points, train_set=train_points
            )
        )
        return pd.DataFrame(
            [mat[1].numpy() for mat in [e[1] for e in influences][0]][0]
        ).transpose()

