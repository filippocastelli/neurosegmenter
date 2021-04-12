from argparse import ArgumentParser
from pathlib import Path


from neuroseg.config import PredictConfig
from neuroseg.tiledpredict import DataPredictor
from neuroseg.performance_eval import PerformanceEvaluator

from neuroseg.descriptor import RunDescriptorLight

def main(cfg_path):

    config = PredictConfig(cfg_path)
    # setup_logger(config.logfile_path)
    predict(config)

def predict(predict_config: PredictConfig):
    dp = DataPredictor(predict_config)
    ev = PerformanceEvaluator(predict_config, dp.predicted_data)
    performance_dict = ev.measure_dict
    _ = RunDescriptorLight(predict_config, performance_metrics_dict=performance_dict)
    return performance_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/tests/test_predict_cfg_3d.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    cfg_path = Path(args.configuration_path_str)
    
    main(cfg_path)