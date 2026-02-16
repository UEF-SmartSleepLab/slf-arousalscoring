import keras
import mlflow
import tensorflow as tf

from datetime import datetime as dt
from functools import partial
from pathlib import Path
from tensorflow.keras import backend as K
from typing import Any

# Enable operation determinism
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


def get_model(cfg: dict[str, Any]):
    """Import and load the model."""
    import slf_simultscoring.utime_tf.func_models as models
    return models.UTime(
        input_names=cfg['input_names'],
        block_args=cfg['block_args'],
        output_args=[oarg for oarg in cfg['output_args'] if oarg['output_name'] in cfg['output_names']],
        aspp_depth=cfg['aspp_depth'],
        activation=eval(cfg['activation'])
    )


def dict_to_io(d, input_names=[], output_names=[], n_classes=None):
    inputs = {k: d[k] for k in input_names}
    if n_classes is not None:
        outputs = {k: tf.one_hot(d[k], depth=n_classes[k]) for k in output_names}
    else:
        outputs = {k: tf.expand_dims(d[k], -1) for k in output_names}

    return inputs, outputs


def training_loop(
        train_config: dict[str, Any],
        model_config: dict[str, Any],
        train_datasets: dict[str, tf.data.Dataset],
        val_datasets: dict[str, tf.data.Dataset],
        one_hot: bool = False,
        seed: int = 42) -> keras.Model:
    """Train the 1d unet model using tensorflow.""" 
    train_start = dt.now().strftime('%Y%m%d_%H%M%S')

    # There may be multiple training and validation sets.
    # For now, use only the splits named 'val' and 'train'
    val_ds = val_datasets['val']
    val_size = len(val_ds)
    train_ds = train_datasets['train']
    train_size = len(train_ds)

    if one_hot:
        io_map_func = partial(dict_to_io,
            input_names=model_config['input_names'],
            output_names=model_config['output_names'],
            n_classes={oarg['output_name']: oarg['n_classes'] for oarg in model_config['output_args'] if oarg['output_name'] in model_config['output_names']})
    else:
        io_map_func = partial(dict_to_io,
            input_names=model_config['input_names'],
            output_names=model_config['output_names'])

    train_ds = (train_ds.map(io_map_func)
        .repeat()
        .shuffle(buffer_size=5*train_config['batch_size'], reshuffle_each_iteration=True)
        .batch(train_config['batch_size'])
        .prefetch(tf.data.AUTOTUNE)
    )

    mlflow_log_cb = LogMLflowMetrics()
    scheduler_cb = OneCycleScheduler(
        min_lr=train_config['peak_learning_rate'] / train_config['lr_div_factor'],
        max_lr=train_config['peak_learning_rate'],
        epochs=train_config['epochs'],
        after_cycle_epochs=train_config['after_cycle_epochs']
    )
    callbacks = [mlflow_log_cb, scheduler_cb]

    if train_config['model_ckpt_dir'] is None:
        ckpt_dir = '/tmp'
    else:
        ckpt_dir = train_config['model_ckpt_dir']

    # Saving the model in .keras format
    ckpt_fpath = Path(ckpt_dir) / f'best_model_{train_start}.keras'
    mlflow.log_param('ckpt_fpath', ckpt_fpath)
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        str(ckpt_fpath), monitor='val_loss', save_best_only=True)
    callbacks.append(ckpt_cb)    

    val_ds = val_ds.map(io_map_func).batch(1).repeat()

    model = get_model(model_config)
    if 'initial_weights_path' in train_config.keys() and train_config['initial_weights_path'] is not None: # start from pre-trained weights instead of randomly initializing
        model.load_weights(train_config['initial_weights_path'], skip_mismatch=True)  # skip mismatch is true because the number of classes (weights in last layer) can be different
    
    optimizer = keras.optimizers.AdamW()
    if one_hot:
        loss_object = {oname: tf.keras.losses.CategoricalFocalCrossentropy(gamma=train_config['gamma'], alpha=train_config['alpha']) for oname in model_config['output_names']}
        metrics = {oname: keras.metrics.CategoricalAccuracy() for oname in model_config['output_names']}
    else:
        loss_object = {oname: 'sparse_categorical_crossentropy' for oname in model_config['output_names']}
        metrics = {oname: keras.metrics.SparseCategoricalAccuracy() for oname in model_config['output_names']}

    model.compile(optimizer=optimizer, loss=loss_object,
        metrics=metrics,
        jit_compile=False
    )

    train_steps = train_size // train_config['batch_size']

    model.fit(train_ds, steps_per_epoch=train_steps, epochs=train_config['epochs'],
        validation_data=val_ds, validation_steps=val_size, callbacks=callbacks, verbose=1)

    return keras.models.load_model(ckpt_fpath)


def evaluate(
        model: tf.keras.Model,
        test_ds: tf.data.Dataset,
        config: dict[str, Any],
        one_hot: bool = False,
        prefix: str = 'test') -> None:
    """Evaluate a keras mdoel using test_ds."""
    test_size = len(test_ds)
    
    if one_hot:
        io_map_func = partial(dict_to_io,
            input_names=config['input_names'],
            output_names=config['output_names'],
            n_classes={oarg['output_name']: oarg['n_classes'] for oarg in config['output_args'] if oarg['output_name'] in config['output_names']})
    else:
        io_map_func = partial(dict_to_io,
            input_names=config['input_names'],
            output_names=config['output_names'])

    test_ds = (test_ds
        .map(io_map_func)
        .batch(1)
    )

    scores = model.evaluate(test_ds, steps=test_size)
    score_dict = dict(zip(model.metrics_names, scores))
    
    for k, v in score_dict.items():
        mlflow.log_metric(f'{prefix}_{k}', v)


class LogMLflowMetrics(tf.keras.callbacks.Callback):
    """A callback for logging metrics to MLflow with keras models.
    
    By default, metrics and losses returned by standard Keras
    training loop are logged.
    
    If customized metrics need to be logged, their names and functions
    to get them can be defined in `custom_metric_dict`. The functions will be
    given the model and epoch as arguments.
    """
    def __init__(self, monitor='val_loss',
                 custom_metric_dict=None):
        super().__init__()
        self.monitor = monitor
        self.custom_metric_dict = custom_metric_dict
    
    def on_train_begin(self, logs={}):
        self.best_epoch = 0
        self.best_val_loss = None
        
    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            mlflow.log_metric(k, v, step=epoch)
            
        if self.custom_metric_dict is not None:
            for k, f in self.custom_metric_dict.items():
                custom_metric = f(self.model, epoch)
                mlflow.log_metric(k, custom_metric, step=epoch)
        
        val_loss = logs.get(self.monitor)
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        if val_loss < self.best_val_loss:
            self.best_epoch = epoch
            self.best_val_loss = val_loss
            
        try:
            momentum = K.get_value(self.model.optimizer.momentum)
            mlflow.log_metric('momentum', momentum, step=epoch)
        except AttributeError:
            pass
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        mlflow.log_metric('learning_rate', lr, step=epoch)
        mlflow.log_metric('current_epoch', epoch)
            
    def on_train_end(self, logs={}):
        mlflow.log_metric("best_val_loss", self.best_val_loss)
        mlflow.log_metric("best_epoch", self.best_epoch)


class OneCycleScheduler(tf.keras.callbacks.Callback):
    """Implement one cycle scheduler according to
    https://arxiv.org/pdf/1803.09820.pdf
    
    Args:
        min_lr: The minimum learning rate for the cycle
        max_lr: The max lr for the cycle
        epochs: Total number of epochs
        after_cycle_epochs: The number of epochs to run after the cycle
        after_cycle_decay: The lr decay to use after the cycle
    """
    def __init__(self, min_lr, max_lr, epochs,
                 after_cycle_epochs=20,
                 after_cycle_decay=0.9,
                 max_momentum=0.9,
                 min_momentum=0.8):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        self.epochs = epochs
        self.peak_epoch = int((epochs-after_cycle_epochs) / 2)
        self.lr_step = (max_lr-min_lr) / self.peak_epoch
        self.momentum_step = (max_momentum-min_momentum) / self.peak_epoch
        self.after_cycle_epochs = after_cycle_epochs
        self.after_cycle_decay = after_cycle_decay
        
    def on_train_begin(self, logs=None):
        self.model.optimizer.learning_rate = self.min_lr
        try:
            self.model.optimizer.momentum = self.max_momentum
            self.momentum = True
        except AttributeError:
            # Store not having momentum for further usage
            self.momentum = False
        
    def get_lr(self, epoch):
        """Compute the learning rate."""
        lr = self.model.optimizer.learning_rate
        if epoch < self.peak_epoch:
            lr = lr + self.lr_step
        elif epoch < self.epochs - self.after_cycle_epochs - 1:
            lr = lr - self.lr_step
        else:
            lr = lr * self.after_cycle_decay
        return lr
    
    def get_momentum(self, epoch):
        """Compute the momentum."""
        momentum = self.model.optimizer.momentum
        if epoch < self.peak_epoch:
            momentum = momentum - self.momentum_step
        elif epoch < self.epochs - self.after_cycle_epochs - 1:
            momentum = momentum + self.momentum_step
        else:
            momentum = self.max_momentum
        return momentum
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.optimizer.learning_rate = self.get_lr(epoch)
        if self.momentum:
            self.model.optimizer.momentum = self.get_momentum(epoch)
