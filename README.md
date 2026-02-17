# slf-arousalscoring
Train and evaluate PSG arousal scoring models using [sleeplab-fomat](https://github.com/UEF-SmartSleepLab/sleeplab-format).

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
git clone <new repo url>
```

You can copy the URL for cloning from Code -> Clone. Then, create a new python environment with your tool of choice (e.g conda or venv) and activate the environment. After that, install the dependencies with pip in the project root folder:

```console
pip install --upgrade pip
pip install -e .
```

## Usage

This project uses [MLFlow](https://mlflow.org/) to track the training runs. The MLFlow server can be started for example in port 5001 by running in the project's python environment:

```console
mlflow server --port 5001 --backend-store-uri /tmp/mlruns --default-artifact-root /tmp/mlartifacts
```

If you want to persist the data, substitute the `/tmp` folders with other location. See MLFlow's documentation for more options.

After starting the MLFlow server, model training can be started using a configuration file (examples are under `config_files/`) by running in the project root folder:

```console
python src/slf_arousalscoring/train.py --param-path config_files/tensorflow_MESA_arousals.yml --mlflow-tracking-uri http://localhost:5001
```

To use GPU with Tensorflow, you need to have [CUDA and CUDNN installed properly](https://www.tensorflow.org/install/pip). Then, you can use a GPU for training with `--tf-use-gpu` and `--visible-device` flags, for example to run training on GPU 1:

```console
python src/slf_simultscoring/train.py --tf-use-gpu --visible-device 1 \
    --param-path config_files/tensorflow_MESA_arousals.yml \
    --mlflow-tracking-uri http://localhost:5001
```

You can score arousals using the model by running:

```console
python src/slf_arousalscoring/score_slf_dataset_arousals.py --ds-dir /tmp/data/MESA/MESA_extracted --series-names psg --scorer-ckpt-dir /tmp/models/slf-simultscoring/best_model_20250325_142347.keras --tf-config-path config_files/tensorflow_MESA_arousals.yml --ds-save-dir /tmp/Autoscores --cuda-visible-device 1
```

## License

`slf-arousalscoring` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
