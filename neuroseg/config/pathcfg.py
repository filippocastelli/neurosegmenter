from pathlib import Path

def load_paths(cfg):
    """load paths from cfg, create missing folders, return a path dict"""
    dataset_path  = Path(cfg["dataset_path"])
    train_path = dataset_path.joinpath("train")
    val_path = dataset_path.joinpath("val")
    test_path = dataset_path.joinpath("test")

    out_path = Path(cfg["output_path"])
    tmp_path = out_path.joinpath("tmp")
    logs_path = out_path.joinpath("logs")
    
    # output files
    final_model_path = out_path.joinpath("final_model.hdf5")
    csv_summary_path = out_path.joinpath("run_summary.csv")
    model_history_path = out_path.joinpath("model_history.pickle")
    logfile_path = out_path.joinpath("run_log.log")
    
    path_dict = {
        "dataset": dataset_path,
        "out": out_path,
        "tmp": tmp_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "logs": logs_path,
        "final_model" : final_model_path,
        "model_history" : model_history_path,
        "csv_summary" : csv_summary_path,
        "logfile" : logfile_path
        }
    
    for key, path in path_dict.items():
        if "." not in path.name:
            path.mkdir(exist_ok=True, parents=True)
        
    return path_dict

