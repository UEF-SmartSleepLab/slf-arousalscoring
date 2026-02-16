import tensorflow as tf

import argparse
import logging
import numpy as np
import sleeplab_format as slf
import yaml

from datetime import timedelta
from functools import partial
from pathlib import Path
from slf_simultscoring.train import set_gpu_visibility

logger = logging.getLogger(__name__)


def score_samples_tf(ckpt, inputs, output_names) -> dict[str, np.array]:
    outputs = ckpt(inputs)
    
    return {oname: output.numpy() for oname, output in zip(output_names, outputs)}


def parse_hypnogram(arr, start_ts, scorer, epoch_sec=30.0) -> slf.models.Hypnogram:
    stage_map = {
        0: slf.models.AASMSleepStage.W,
        1: slf.models.AASMSleepStage.N1,
        2: slf.models.AASMSleepStage.N2,
        3: slf.models.AASMSleepStage.N3,
        4: slf.models.AASMSleepStage.R
    }
    annotations = []
    pred = np.argmax(arr, axis=1)
    for i, stage_int in enumerate(pred):
        #print(f'stage_int = {stage_int}')
        start_delta = i * timedelta(seconds=epoch_sec)
        stage = slf.models.Annotation[slf.models.AASMSleepStage](
            name=stage_map[stage_int],
            start_ts=start_ts + start_delta,
            start_sec=float(start_delta.seconds),
            duration=epoch_sec
        )
        annotations.append(stage)

    return slf.models.Hypnogram(annotations=annotations, scorer=scorer)

def classify_arousals(arr: np.array, no_event_threshold: float = 0.5):
    # Define which is higher: arousal or no arousal confidence
    res = np.argmax(arr[..., 0:], axis=1)
    # Then, set the indices to one where the confidence of arousal is higher than the event threshold
    res[arr[..., 1] > 1-no_event_threshold] = 1

    return res

def get_continuous_segments(arr, val_arr=[1], min_event_length=3):
    """Compute start and stop idx of each segment in samples belonging to val_arr."""
    
    segs = []
    event_on = False
    stop = -11
    
    if min_event_length is None:
        min_event_length = 1
    
    for i, v in enumerate(arr):
        if v in val_arr:
            if not event_on:
                start = i
                event_on = True
        else:
            if event_on:
                if not np.any(np.isin(arr[max(0, i):i], val_arr)):
                    event_on = False

                    # Discard arousal starting less than 10 s after previous arousal
                    if start - stop - 1 < 10:
                        Discard = True
                    else:
                        Discard = False

                    stop = i

                    if not Discard and stop - start >= min_event_length:
                        etype = slf.models.AASMEvent.AROUSAL
                        segs.append((start, stop, etype))

    return segs

def parse_arousals(arr, start_ts, scorer, no_event_threshold) -> slf.models.AASMEvents:
    arousals = get_continuous_segments(classify_arousals(arr, no_event_threshold=no_event_threshold), min_event_length=3)
    annotations = []
    
    for start, stop, etype in arousals:
        arousal = slf.models.Annotation[slf.models.AASMEvent](
            name=etype,
            start_ts=start_ts + timedelta(seconds=start),
            start_sec=float(start),
            duration = float(stop - start)
        )
        annotations.append(arousal)

    return slf.models.AASMEvents(annotations=annotations, scorer=scorer)


def parse_annotations(scorer_outputs, start_ts, scorer='utime', no_event_threshold=0.5) -> slf.models.BaseAnnotations:
    res = {}
    if 'hypnogram' in scorer_outputs.keys():
        res[f'{scorer}_hypnogram'] = parse_hypnogram(scorer_outputs['hypnogram'], start_ts, scorer)
    
    if 'arousals' in scorer_outputs.keys():
        res[f'{scorer}_aasmarousals'] = parse_arousals(scorer_outputs['arousals'], start_ts, scorer, no_event_threshold=no_event_threshold)

    return res


def get_sds_config(model_config, ds_config, ds_dir, series_name, fs, input_name_map=None):
    """Create a sleeplab-tf-dataset configuration for scorer inputs."""
    import sleeplab_tf_dataset as sds
    cfg_dict = {
        'ds_dir': ds_dir,
        'series_name': series_name,
        'start_sec': 0, # use whole-night signals
        'duration': -1.0,
        'roi_src_type': ds_config['roi_src_type'],
        'roi_src_name': ds_config['roi_src_name'],

        'components': {
            k: {
                'src_name': k if input_name_map is None else input_name_map[k],
                'ctype': 'sample_array',
                'fs': fs
            } for k in model_config['input_names']
        }
    }
    return sds.config.DatasetConfig.model_validate(cfg_dict)


def score_slf_ds(
        ds_dir: Path,
        batch_size: int,
        scorer_ckpt_dir: Path,
        series_names: list[str],
        scorer_name: str = 'utime',
        no_event_threshold: float = 0.5,
        rng_seed: int = 42,
        tf_config_path: Path | None = None,
        input_name_map: dict[str, str] | None = None) -> slf.models.Dataset:
    """Score with automatic model."""
    import sleeplab_tf_dataset as sds

    with open(tf_config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model_config = cfg['model']
    ds_config = [cfg['datasets'][ds_name] for ds_name in cfg['datasets']][0]
        
    scorer_ckpt = tf.keras.models.load_model(scorer_ckpt_dir)
    score_func = partial(score_samples_tf, output_names=model_config['output_names'])
    
    logger.info(f'Reading SLF dataset from {ds_dir}...')
    slf_ds = slf.reader.read_dataset(ds_dir, series_names)

    annot_series = {}

    for series_name in series_names:
        logger.info(f'Scoring series {series_name}...')
        nsamples = len(slf_ds.series[series_name].subjects)
        assert nsamples % batch_size == 0
        nbatches = nsamples // batch_size
        first_input_key = model_config['input_names'][0]
        first_input_name = first_input_key if input_name_map is None else input_name_map[first_input_key]
        fs = (list(slf_ds.series[series_name]
            .subjects.values())[0]
            .sample_arrays[first_input_name]
            .attributes.sampling_rate
        )
        cfg = get_sds_config(model_config, ds_config, ds_dir, series_name, fs, input_name_map=input_name_map)
        tf_ds = sds.dataset.from_slf_dataset(slf_ds=slf_ds, cfg=cfg)
        ds_iter = tf_ds.batch(batch_size).as_numpy_iterator()

        subjects = {}
        subject_ids = list(slf_ds.series[series_name].subjects.keys())

        for i in range(nbatches):
            xs = ds_iter.next()
            logger.info(f'Batch {i}: scoring samples...')
            scorer_outputs = score_func(scorer_ckpt, xs)

            for j in range(batch_size):
                # from_slf_dataset preserves the order of subject keys, so we can assume that the
                # ds_iter yields subjects in this order.
                subject_idx = i*batch_size + j
                subject_id = subject_ids[subject_idx]
                logger.info(f'Parsing annotations for subject ID {subject_id}...')
                _subj = slf_ds.series[series_name].subjects[subject_id]
                
                annotations = parse_annotations(
                    scorer_outputs=scorer_outputs, #={k: np.array(v[j]) for k, v in scorer_outputs.items()},
                    start_ts=_subj.metadata.recording_start_ts,
                    scorer=scorer_name,
                    no_event_threshold=no_event_threshold
                )
                subjects[subject_id] = slf.models.Subject(
                    metadata=_subj.metadata,
                    annotations=annotations)

        # Return a SLF dataset with only the annotations
        annot_series[series_name] = slf.models.Series(name=series_name, subjects=subjects)

    ds = slf.models.Dataset(name=slf_ds.name, series=annot_series)

    return ds

def map_input_names(tf_config_path: Path | None = None):
    with open(tf_config_path, 'r') as f:
        config = yaml.safe_load(f)
    ds_name = [ds for ds in config['datasets']][0] #todo: support for multiple datasets
    
    input_name_map = {input_name: config['datasets'][ds_name]['components'][input_name]['src_name'] for input_name in config['model']['input_names']}

    return input_name_map

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds-dir', type=str, required=True,
        help='Folder where the SLF dataset is')
    parser.add_argument('--series-names', type=str, nargs='*', required=True,
        help='The name of the created SLF series')
    parser.add_argument('--scorer-ckpt-dir', type=str, required=True,
        help='Path to the automatic scoring model checkpoint')
    parser.add_argument('--ds-save-dir', type=str,
        help='If given, save the results to this location instead of the --ds-dir')
    parser.add_argument('--tf-config-path', type=str,
        help='Path to the tensorflow simultscoring config file, if Tensorflow model is used')
    parser.add_argument('--scorer-name', type=str, default='utime',
        help='scorer name for the annotations')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
        help='batch size for generating and scoring samples')
    parser.add_argument('--restore-ckpt-func', default='slf_simultscoring.utime_flax.training.restore_checkpoint')
    parser.add_argument('--no-event-threshold', type=float, default=0.5,
        help='minimum confidence to score "no event"')
    parser.add_argument('--cuda-visible-device', type=str, default='0')

    return parser


def run_cli() -> None:
    parser = get_parser()
    args = parser.parse_args()

    set_gpu_visibility(args.cuda_visible_device, tf_cpu_only=False)
    input_name_map = map_input_names(tf_config_path=args.tf_config_path)

    annotation_ds = score_slf_ds(
        ds_dir=Path(args.ds_dir),
        batch_size=args.batch_size,
        scorer_ckpt_dir=Path(args.scorer_ckpt_dir),
        tf_config_path=args.tf_config_path,
        series_names=args.series_names,
        scorer_name=args.scorer_name,
        restore_ckpt_func_str=args.restore_ckpt_func,
        no_event_threshold=args.no_event_threshold,
        input_name_map=input_name_map)

    if args.ds_save_dir is not None:
        ds_save_dir = Path(args.ds_save_dir)
        annotation_ds.name = ds_save_dir.name
    else:
        ds_save_dir = Path(args.ds_dir)

    slf.writer.write_dataset(annotation_ds, basedir=Path(ds_save_dir).parent)
    logger.info('Autoscoring finished.')


if __name__ == '__main__':
    run_cli()
