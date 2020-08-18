=================
API Documentation
=================

This is the API documentation for ``scikit-ika``.

Data: :mod:`data`
=================

.. automodule:: skika.data
   :no-members:
   :no-inherited-members:

.. currentmodule:: skika

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   data.stream_generator
   data.reccurring_concept_stream
   data.wind_sim_generator
   data.bernoulli_stream
   data.generate_dataset
   data.hyper_plane_generator_redund
   data.random_rbf_generator_redund
   data.stream_generator_redundancy_drift

Evaluation: :mod:`evaluation`
=============================

.. automodule:: skika.evaluation
   :no-members:
   :no-inherited-members:

.. currentmodule:: skika

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   evaluation.inspect_recurrence

Visualisation: :mod:`visualisation`
===================================

.. automodule:: skika.visualisation
   :no-members:
   :no-inherited-members:

.. currentmodule:: skika

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   visualisation.wind_sim_feature_visualisation
   visualisation.wind_sim_visualisation

Hyper-parameter tuning: :mod:`hyper_parameter_tuning`
===================================

.. automodule:: skika.hyper_parameter_tuning
   :no-members:
   :no-inherited-members:

.. currentmodule:: skika

.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   hyper_parameter_tuning.drift_detectors.build_pareto_knowledge_drifts
   hyper_parameter_tuning.drift_detectors.evaluate_drift_detection_experiment
   hyper_parameter_tuning.drift_detectors.evaluate_drift_detection_knowledge
   hyper_parameter_tuning.trees_arf.build_pareto_knowledge_trees
   hyper_parameter_tuning.trees_arf.evaluate_prequential_and_adapt
   hyper_parameter_tuning.trees_arf.meta_feature_generator