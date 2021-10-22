"""
Contains basic hyperparameter optimizations.
"""
import numpy as np
import tensorflow as tf
import os
import itertools
import tempfile
import shutil
import collections
import logging
from functools import reduce
from operator import mul
from typing import Dict, Iterable, List, Optional
import deepchem

from deepchem.data import Dataset
from deepchem.models.models import Model
from deepchem.trans import Transformer
from deepchem.metrics import Metric
from deepchem.models import ValidationCallback, callbacks
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename

logger = logging.getLogger(__name__)


class GridHyperparamOpt(HyperparamOpt):
  """
  Provides simple grid hyperparameter search capabilities.

  This class performs a grid hyperparameter search over the specified
  hyperparameter space. This implementation is simple and simply does
  a direct iteration over all possible hyperparameters and doesn't use
  parallelization to speed up the search.

  Examples
  --------
  This example shows the type of constructor function expected.

  >>> import sklearn
  >>> import deepchem as dc
  >>> optimizer = dc.hyper.GridHyperparamOpt(lambda **p: dc.models.GraphConvModel(**p))

  Here's a more sophisticated example that shows how to optimize only
  some parameters of a model. In this case, we have some parameters we
  want to optimize, and others which we don't. To handle this type of
  search, we create a `model_builder` which hard codes some arguments
  (in this case, `n_tasks` and `n_features` which are properties of a
  dataset and not hyperparameters to search over.)

  >>> def model_builder(model_dir, **model_params):
  ...   n_layers = model_params['layers']
  ...   layer_width = model_params['width']
  ...   dropout = model_params['dropout']
  ...   return dc.models.MultitaskClassifier(
  ...     n_tasks=5,
  ...     n_features=100,
  ...     layer_sizes=[layer_width]*n_layers,
  ...     dropouts=dropout
  ...   )
  >>> optimizer = dc.hyper.GridHyperparamOpt(model_builder)

  """

  def hyperparam_search(
      self,
      params_dict: Dict,
      train_dataset: Dataset,
      valid_dataset: Dataset,
      metric: Metric,
      output_transformers: List[Transformer] = [],
      nb_epoch: int = 10,
      use_max: bool = True,
      replicates: int = 1,
      rng_seeds: Optional[List[int]] = None,
      restore_best_checkpoint: bool = False,
      checkpoint_interval: int = 10,
      logdir: Optional[str] = None,
      **kwargs,
  ):
    """Perform hyperparams search according to params_dict.

    Each key to hyperparams_dict is a model_param. The values should
    be a list of potential values for that hyperparam.

    Parameters
    ----------
    params_dict: Dict
      Maps hyperparameter names (strings) to lists of possible
      parameter values.
    train_dataset: Dataset
      dataset used for training
    valid_dataset: Dataset
      dataset used for validation(optimization on valid scores)
    metric: Metric
      metric used for evaluation
    output_transformers: list[Transformer]
      Transformers for evaluation. This argument is needed since
      `train_dataset` and `valid_dataset` may have been transformed
      for learning and need the transform to be inverted before
      the metric can be evaluated on a model.
    nb_epoch: int, (default 10)
      Specifies the number of training epochs during each iteration of optimization.
      Not used by all model types.
    use_max: bool, optional
      If True, return the model with the highest score. Else return
      model with the minimum score.
    replicates: int, optional
      How many times to evaluate each hyperparameter combination (for variance estimation)
    rng_seeds: list[int], optional
      If provided, will set the global rng seeds of numpy and tf prior to training each 
      model replicate. Must be the same size as the number of replicates and all the
      elements must be unique! Used for getting reproducible results.
    restore_best_checkpoint: bool, optional
      Restores the model from the best checkpoint achieved during the training run for final
      evaluation
    checkpoint_interval: int, optional
      Used only if restore_best_checkpoint is set to True. Determines the frequency at which
      checkpoints are created when using the validation callback.
    logdir: str, optional
      The directory in which to store created models. If not set, will
      use a temporary directory.
    kwargs: remaining kwargs are passed to the .fit() method of the estimator

    Returns
    -------
    Tuple[`best_model`, `best_hyperparams`, `all_scores`]
      `(best_model, best_hyperparams, all_scores)` where `best_model` is
      an instance of `dc.model.Model`, `best_hyperparams` is a
      dictionary of parameters, and `all_scores` is a dictionary mapping
      string representations of hyperparameter sets to lists of validation
      scores.
    """
    # check argument validities
    hyperparams = params_dict.keys()
    hyperparam_vals = params_dict.values()
    for hyperparam_list in params_dict.values():
      assert isinstance(hyperparam_list, collections.abc.Iterable)
    assert isinstance(replicates, int) and replicates > 0, "replicates must be a positive integer"
    if rng_seeds is not None:
      assert isinstance(rng_seeds, Iterable) and len(rng_seeds) == replicates, "number of passed rng seeds must be \
        equal to the number of replicates"
      for seed in rng_seeds:
        assert isinstance(seed, int) and seed >= 0, "rng seed must be a non-negative integer"
      assert len(set(rng_seeds)) == replicates, "all rng seeds must be unique"
    
    number_combinations = reduce(mul, [len(vals) for vals in hyperparam_vals])

    if use_max:
      best_validation_score = -np.inf
    else:
      best_validation_score = np.inf
    best_hyperparams = None
    all_scores = {}
    
    # set training kwargs used by all models
    train_kwargs = {
      "valid_dataset": valid_dataset,
      "metric": metric,
      "output_transformers": output_transformers,
      "nb_epoch": nb_epoch,
      "use_max": use_max,
      "restore_best_checkpoint": restore_best_checkpoint,
      "checkpoint_interval": checkpoint_interval
    }
    # loop through HPs, build and test models
    for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparam_vals)):
      logger.info("Fitting model %d/%d" % (ind + 1, number_combinations))
      model_dir = self.create_model_dir(logdir=logdir)
      # Construction dictionary mapping hyperparameter names to values
      model_params = self.build_model_params_dict(hyperparams, hyperparameter_tuple)
      logger.info("hyperparameters: %s" % str(model_params))
      # set model directory
      model_params['model_dir'] = model_dir
      
      # build and evaluate model replicates
      hyper_params = dict(zip(hyperparams, hyperparameter_tuple))
      hp_str = _convert_hyperparam_dict_to_filename(hyper_params)
      valid_scores = []
      for i in range(replicates):
        logger.info("Fitting replicate %i" % (i + 1))
        # check for rng seeds and set
        rng_seed = rng_seeds[i] if rng_seeds is not None else None
        # build and train a model
        model = self.train_model(model_params, train_dataset, rng_seed=rng_seed, **train_kwargs)
        # evaluate trained model
        multitask_scores = model.evaluate(valid_dataset, [metric],
                                          output_transformers)
        valid_scores.append(multitask_scores[metric.name])

      all_scores[hp_str] = valid_scores
      mean_val_score = np.mean(valid_scores)
      if (use_max and mean_val_score >= best_validation_score) or (
          not use_max and mean_val_score <= best_validation_score):
        best_validation_score = mean_val_score
        best_hyperparams = hyperparameter_tuple
      logger.info("Model %d/%d, Metric %s, Validation set %s: %f" %
                  (ind + 1, number_combinations, metric.name, ind, mean_val_score))
      logger.info("\tbest_validation_score so far: %f" % best_validation_score)
      # clean up: remove model dir
      shutil.rmtree(model_dir)

    if best_hyperparams is not None:
      # retrain best model
      logger.info("Re-training best model: %s" % str(best_hyperparams))
      model_dir = self.create_model_dir(logdir=logdir)
      model_params = self.build_model_params_dict(hyperparams, best_hyperparams)
      model_params['model_dir'] = model_dir
      rng_seed = rng_seeds[0] if rng_seeds is not None else None
      best_model = self.train_model(model_params, train_dataset, rng_seed=rng_seed, **train_kwargs)
      # evaluate best model
      train_scores = best_model.evaluate(train_dataset, [metric],
                                            output_transformers)
      train_score = train_scores[metric.name]
      valid_scores = best_model.evaluate(valid_dataset, [metric],
                                            output_transformers)
      valid_score = valid_scores[metric.name]
      logger.info("Best hyperparameters: %s" % str(best_hyperparams))
      logger.info("train_score: %f" % train_score)
      logger.info("validation_score: %f" % valid_score)
      # clean up: remove model dir
      shutil.rmtree(model_dir)
      return best_model, best_hyperparams, all_scores
    else:
      logging.warning("Failed to sucessfully train any models")
      return None, None, None

  @staticmethod
  def create_model_dir(logdir:str=None) -> str:
    if logdir is not None:
      # model_dir = os.path.join(logdir, str(ind))
      model_dir = logdir
      logger.info("model_dir is %s" % model_dir)
      try:
        os.makedirs(model_dir)
      except OSError:
        if not os.path.isdir(model_dir):
          logger.info("Error creating model_dir, using tempfile directory")
          model_dir = tempfile.mkdtemp()
    else:
      model_dir = tempfile.mkdtemp()
    return model_dir

  @staticmethod
  def build_model_params_dict(hyperparams: Iterable, hyperparameter_tuple: Iterable) -> dict:
    model_params = {}
    for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
      model_params[hyperparam] = hyperparam_val
    return model_params

  def train_model(self, model_params:dict, train_dataset:Dataset, rng_seed:int=None, 
    restore_best_checkpoint:bool=False, valid_dataset:Dataset=None, checkpoint_interval:int=10,
    metric:Metric=None, use_max:bool=True, nb_epoch:int=10,
    output_transformers=[]) -> Model:
    """Build and train a single model"""
    if rng_seed is not None:
      np.random.seed(rng_seed)
      tf.random.set_seed(rng_seed)
      # TODO: set pytorch rng seed as well!

    if restore_best_checkpoint:
      val_dir = tempfile.mkdtemp()
      val_callback = ValidationCallback(valid_dataset, checkpoint_interval, [metric], save_dir=val_dir, 
        save_on_minimum=(not use_max), transformers=output_transformers)
    else:
      val_dir, val_callback = None, []

    # build and train the model
    model = self.model_builder(**model_params)
    # mypy test throws error, so ignoring it in try
    try:
      model.fit(train_dataset, nb_epoch=nb_epoch, callbacks=val_callback)  # type: ignore
    # Not all models have nb_epoch or callbacks
    except TypeError:
      model.fit(train_dataset)
    try:
      model.save()
    # Some models autosave
    except NotImplementedError:
      pass
    # for models with checkpointing, restore the latest (best) checkpoint if enabled
    if restore_best_checkpoint:
      try:
        model.restore(model_dir=val_dir)
      except NotImplementedError:
        logger.warning("Could not restore model weights from validation checkpoint. \
          Using latest weights instead")
        pass
    # clean up: remove validation dir if used
    if (val_dir is not None) and os.path.exists(val_dir):
      shutil.rmtree(val_dir)
    return model

