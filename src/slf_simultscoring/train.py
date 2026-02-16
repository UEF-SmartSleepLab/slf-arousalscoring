import argparse
import logging
import os
import mlflow
import tempfile
import tensorflow as tf
import yaml

from importlib import import_module
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def set_gpu_visibility(gpu_device: str, tf_cpu_only: bool = True, num_tf_threads: int = 16):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    tf.config.threading.set_inter_op_parallelism_threads(num_tf_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_tf_threads)

    # Restrict tensorflow not to use GPU
    if tf_cpu_only:
        tf.config.experimental.set_visible_devices([], 'GPU')


def log_config_to_mlflow(cfg: dict[str, Any]):
    with tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / 'config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(cfg, f)
            mlflow.log_artifact(f.name)


def train(config: dict[str, Any], mlflow_config: dict[str, Any], seed: int = 42) -> None:
    """Train a model according to the config."""
    tf.random.set_seed(seed)
    # This needs to be imported after setting visible devices
    import sleeplab_tf_dataset as sds
    
    # Import the model submodule
    module = import_module(config['module_str'])

    # Delete unnecessary items from ds config before loading everything
    for ds in config['datasets']:
        for item in config['datasets'][ds]['components'].copy():
            if item not in config['model']['input_names'] and item not in config['model']['output_names']:
                config['datasets'][ds]['components'].pop(item)
    
    # Instantiate the datasets
    logger.info('instantiating the datasets...')
    datasets = sds.compose.load_split_concat(config['datasets'], seed=seed)

    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    mlflow.set_experiment(mlflow_config['experiment'])

    # Run the training loop
    logger.info('running the training loop...')
    with mlflow.start_run(mlflow_config['run_id']):
        log_config_to_mlflow(config)

        # Support multiple training sets
        train_keys = [k for k in datasets.keys() if k.startswith('train')]
        train_datasets = {k: datasets[k] for k in train_keys}

        # Support multiple validation sets
        val_keys = [k for k in datasets.keys() if k.startswith('val')]
        val_datasets = {k: datasets[k] for k in val_keys}

        # Check if one hot encoding is required
        if 'alpha' in config['training'].keys() and config['training']['alpha'] is not None:
            one_hot = True
        else:
            one_hot = False

        state = module.training.training_loop(
            config['training'], config['model'], train_datasets, val_datasets, one_hot=one_hot)

        logger.info('Evaluating the model with test sets...')

        # Support multiple test sets
        test_keys = [k for k in datasets.keys() if k.startswith('test')]
        for test_key in test_keys:
            logger.info(f'Evaluating model on {test_key}...')
            module.training.evaluate(state, datasets[test_key], config['model'], one_hot=one_hot, prefix=test_key)

    logger.info('training done.')


def get_parser() -> argparse.ArgumentParser:
    """Create an ArgumentParser to parse the cli args.
    
    A --module, or -m argument is mandatory
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--param-path', type=str, required=True,
        help='full path to the parameter yaml file')
    parser.add_argument('--mlflow-run-id', type=str, default=None,
        help='use a specific MLflow run if given')
    parser.add_argument('--mlflow-experiment', type=str, default='slf-simultscoring',
        help='use a specific MLflow experiment if given')
    parser.add_argument('--mlflow-tracking-uri', type=str,
        default='http://localhost:5000',
        help='use a specific MLflow tracking URI if given')
    parser.add_argument('--visible-device', type=str, default='1',
        help='The cuda device to which the computation is restricted.')
    parser.add_argument('--tf-use-gpu', action='store_true',
        help='Use this flag if tensorflow should use GPU')

    return parser


def get_config(
        # module_str: str,
        param_path: str,
        mlflow_run_id: str,
        mlflow_experiment: str,
        mlflow_tracking_uri: str,
        # param_path: str | None = None) -> tuple[ConfigDict, ConfigDict]:
    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an ml_collections.ConfigDict according to cli args.
    
    Returns:
        a ConfigDict of data and training related configurations
        a dict of MLflow related configurations to separate
            MLflow control logic from other training logic
    """
    # Read the yaml config file
    with open(param_path, 'r') as f:
        config = yaml.safe_load(f)

    # Construct the MLflow config
    mlflow_config = {
        'run_id': mlflow_run_id,
        'experiment': mlflow_experiment,
        'tracking_uri': mlflow_tracking_uri
    }

    return config, mlflow_config 


def run_cli() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_gpu_visibility(args.visible_device, tf_cpu_only=not(args.tf_use_gpu))

    config, mlflow_config = get_config(
        args.param_path,
        args.mlflow_run_id, args.mlflow_experiment, args.mlflow_tracking_uri)    
    
    train(config, mlflow_config)


if __name__ == '__main__':
    run_cli()
