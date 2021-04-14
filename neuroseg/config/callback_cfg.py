from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
    )
# TODO: WANDB

class CallbackConfigurator:
    def __init__(self,
                 config):
        self.config = config
        self.callback_names = self.config.callbacks
        self.callback_cfg = self.config.callbacks_cfg
        self.callbacks = self._compile_callback_list()
        
    @staticmethod
    def _get_callback(callback_name):
        SUPPORTED_CALLBACKS = {
            "checkpoint": ModelCheckpoint,
            "csvlogger": CSVLogger,
            "reducelronplateau": ReduceLROnPlateau,
            "tensorboard": TensorBoard
            }
        if callback_name in SUPPORTED_CALLBACKS:
            return SUPPORTED_CALLBACKS[callback_name]
        else:
            raise NotImplementedError(callback_name)
                
    def _compile_callback_list(self):
        callback_list = []
        for callback_name in self.callback_names:
            callback_cls = self._get_callback(callback_name)
            callback_args = self._get_callback_args(callback_name)
            callback = callback_cls(**callback_args)
            callback_list.append(callback)
        return callback_list
    
    def _get_callback_args(self, callback_name):
        config_args = self.callback_cfg[callback_name]
        extra_args = {}
        if callback_name == "checkpoint":
            extra_args["filepath"] = str(self.config.output_path) + "/weights.{epoch:02d}.hdf5"
        elif callback_name == "csvlogger":
            extra_args["filename"] = str(self.config.csv_summary_path)
        elif callback_name == "tensorboard":
            extra_args["log_dir"] = str(self.config.logs_path)
        
        config_args.update(extra_args)
        return config_args
        
        
        
        