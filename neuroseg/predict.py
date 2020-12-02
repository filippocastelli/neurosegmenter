from argparse import ArgumentParser
from pathlib import Path


from config import PredictConfig
from tiledpredict import DataPredictor2D
from performance_eval import PerformanceEvaluator

from descriptor import RunDescriptorLight



def main(cfg_path):

    config = PredictConfig(cfg_path)
    # setup_logger(config.logfile_path)
    dp = DataPredictor2D(config)
    
    ev = PerformanceEvaluator(config, dp.predicted_data)
    performance_dict = ev.measure_dict
    
    _ = RunDescriptorLight(config, performance_metrics_dict=performance_dict)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("-c","--conf", action="store", type=str,
                        dest="configuration_path_str",
                        default="/home/phil/repos/neuroseg/neuroseg/tests/test_predict_cfg.yml",
                        help="Configuration file path")
    
    args, unknown = parser.parse_known_args()
    
    cfg_path = Path(args.configuration_path_str)
    
    main(cfg_path)