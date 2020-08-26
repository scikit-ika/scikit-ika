import os
import numpy as np
from skika.hyper_parameter_tuning.trees_arf.build_pareto_knowledge_trees import BuildTreesKnowledge

def test_build_pareto_knowledge_trees():
    expected_config = [[0.0, 'ARF60'], [0.1, 'ARF30'], [0.2, 'ARF70'], [0.3, 'ARF70'], [0.4, 'ARF60'],
                       [0.5, 'ARF70'], [0.6, 'ARF60'], [0.7, 'ARF30'], [0.8, 'ARF30'], [0.9, 'ARF30']]

    names = ['ARF10','ARF30','ARF60','ARF70','ARF90','ARF100','ARF120','ARF150','ARF200']
    perc_redund = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


    output_dir = os.getcwd()
    name_file = os.sep.join(['.','recurrent-data','hyper-param-tuning','ExamplesTreesKnowledge','Results10-200.csv'])

    paretoBuild = BuildTreesKnowledge(results_file=name_file, list_perc_redund=perc_redund, list_models=names, output=output_dir, verbose=False)
    paretoBuild.load_drift_data()
    paretoBuild.calculate_pareto()


    res = paretoBuild.best_config

    assert np.alltrue(res == expected_config)
