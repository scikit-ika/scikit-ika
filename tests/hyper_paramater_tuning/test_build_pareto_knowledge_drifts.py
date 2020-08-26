import os
import numpy as np
from skika.hyper_parameter_tuning.drift_detectors.build_pareto_knowledge_drifts import BuildDriftKnowledge

def test_build_pareto_knowledge_drifts():
    expected_config = [['PH1', 'ADWIN1', 'DDM3', 'SeqDrift21'],  ['PH11', 'ADWIN1', 'DDM6', 'SeqDrift26'], ['PH11', 'ADWIN3', 'DDM6', 'SeqDrift28'],
                       ['PH15', 'ADWIN3', 'DDM1', 'SeqDrift28'], ['PH15', 'ADWIN5', 'DDM9', 'SeqDrift29'],    ['PH15', 'ADWIN5', 'DDM8', 'SeqDrift29'],
                       ['PH12', 'ADWIN6', 'DDM8', 'SeqDrift28'],  ['PH12', 'ADWIN7', 'DDM4', 'SeqDrift29'], ['PH13', 'ADWIN2', 'DDM4', 'SeqDrift217'],
                       ['PH10', 'ADWIN1', 'DDM4', 'SeqDrift210'], ['PH10', 'ADWIN2', 'DDM4', 'SeqDrift210'], ['PH10', 'ADWIN1', 'DDM4', 'SeqDrift21'],
                       ['PH1', 'ADWIN9', 'DDM1', 'SeqDrift21'],  ['PH11', 'ADWIN8', 'DDM6', 'SeqDrift218'],  ['PH11', 'ADWIN7', 'DDM6', 'SeqDrift218'],
                       ['PH11', 'ADWIN7', 'DDM1', 'SeqDrift218'], ['PH15', 'ADWIN5', 'DDM9', 'SeqDrift218'], ['PH15', 'ADWIN5', 'DDM8', 'SeqDrift218'],
                       ['PH12', 'ADWIN7', 'DDM8', 'SeqDrift218'], ['PH12', 'ADWIN6', 'DDM4', 'SeqDrift218'],['PH13', 'ADWIN4', 'DDM4', 'SeqDrift216'],
                       ['PH10', 'ADWIN1', 'DDM4', 'SeqDrift210'], ['PH2', 'ADWIN1', 'DDM4', 'SeqDrift210'], ['PH10', 'ADWIN2', 'DDM3', 'SeqDrift210'],
                       ['PH2', 'ADWIN1', 'DDM6', 'SeqDrift218'], ['PH11', 'ADWIN5', 'DDM9', 'SeqDrift218'], ['PH11', 'ADWIN4', 'DDM9', 'SeqDrift218'],
                       ['PH9', 'ADWIN1', 'DDM1', 'SeqDrift218'], ['PH15', 'ADWIN1', 'DDM1', 'SeqDrift210'], ['PH12', 'ADWIN9', 'DDM4', 'SeqDrift210'],
                       ['PH8', 'ADWIN9', 'DDM8', 'SeqDrift29'], ['PH16', 'ADWIN8', 'DDM4', 'SeqDrift218'], ['PH10', 'ADWIN3', 'DDM4', 'SeqDrift217'],
                       ['PH2', 'ADWIN1', 'DDM4', 'SeqDrift210'], ['PH10', 'ADWIN2', 'DDM4', 'SeqDrift211'], ['PH10', 'ADWIN1', 'DDM3', 'SeqDrift211']]

    names_stm = ['BernouW1ME0010','BernouW1ME005095','BernouW1ME00509','BernouW1ME0109','BernouW1ME0108','BernouW1ME0208','BernouW1ME0207','BernouW1ME0307','BernouW1ME0306','BernouW1ME0406','BernouW1ME0506','BernouW1ME05506',
                'BernouW100ME0010','BernouW100ME005095','BernouW100ME00509','BernouW100ME0109','BernouW100ME0108','BernouW100ME0208','BernouW100ME0207','BernouW100ME0307','BernouW100ME0306','BernouW100ME0406','BernouW100ME0506','BernouW100ME05506',
                'BernouW500ME0010','BernouW500ME005095','BernouW500ME00509','BernouW500ME0109','BernouW500ME0108','BernouW500ME0208','BernouW500ME0207','BernouW500ME0307','BernouW500ME0306','BernouW500ME0406','BernouW500ME0506','BernouW500ME05506']

    names_detect = [['PH1','PH2','PH3','PH4','PH5','PH6','PH7','PH8','PH9','PH10','PH11','PH12','PH13','PH14','PH15','PH16'],
                   ['ADWIN1','ADWIN2','ADWIN3','ADWIN4','ADWIN5','ADWIN6','ADWIN7','ADWIN8','ADWIN9'],
                   ['DDM1','DDM2','DDM3','DDM4','DDM5','DDM6','DDM7','DDM8','DDM9','DDM10'],
                   ['SeqDrift21','SeqDrift22','SeqDrift23','SeqDrift24','SeqDrift25','SeqDrift26','SeqDrift27','SeqDrift28','SeqDrift29','SeqDrift210',
                    'SeqDrift211','SeqDrift212','SeqDrift213','SeqDrift214','SeqDrift215','SeqDrift216','SeqDrift217','SeqDrift218']]

    output_dir = os.getcwd()
    directory_path_files = os.sep.join(['.','recurrent-data','hyper-param-tuning','ExampleDriftKnowledge']) # Available in hyper-param-tuning-examples repository

    pareto_build = BuildDriftKnowledge(results_directory=directory_path_files, names_detectors=names_detect, names_streams=names_stm, output=output_dir, verbose =False)
    pareto_build.load_drift_data()
    pareto_build.calculate_pareto()


    res = pareto_build.best_config

    assert np.alltrue(res == expected_config)
