"""Tabular task profiles executed by the shared isolated workflow runner."""
import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .config import load_config
from .workflow import prepare_run, train_run, evaluate_run, analyze_run
from .prediction.inference import predict_csv


def task_settings(config, name, models=None):
    settings = deepcopy(config.get('defaults', {}))
    for section, values in config['profiles'][name].items():
        settings.setdefault(section, {}).update(values)
    if models:
        settings.setdefault('training', {})['models'] = list(models)
    return settings


def build_tabular_plan(config, models=None, tasks=None):
    selected = list(tasks or config['profiles'])
    if not selected or len(set(selected)) != len(selected) or set(selected) - config['profiles'].keys():
        raise ValueError('Tasks must be unique profile names declared in workflow.toml')
    if config.get('steps') or config.get('models'):
        raise ValueError('Tabular overrides belong in defaults/profiles training sections')
    if models is not None and (not models or len(set(models)) != len(models)):
        raise ValueError('Model names must be nonempty and unique')
    plan = []
    for name in selected:
        if not name.replace('_', '').isalnum():
            raise ValueError(f'Invalid task profile name: {name}')
        settings = task_settings(config, name, models)
        task = settings.get('data', {}).get('task')
        candidates = settings.get('training', {}).get('models', ['linear', 'forest'])
        if task not in ('classification', 'regression'):
            raise ValueError(f'{name}: data.task must be classification or regression')
        if not isinstance(candidates, list) or not candidates or any(m not in ('linear', 'forest', 'svm', 'svm_linear', 'knn', 'gnb', 'xgboost') for m in candidates):
            raise ValueError(f'{name}: invalid training.models')
        if task == 'regression' and 'gnb' in candidates:
            raise ValueError('gnb only supports classification; select --tasks classification or different models')
        stages = ['prepare']
        if settings.get('analysis', {}).get('methods'):
            stages.append('analyze')
        stages.extend(['train', 'evaluate', 'predict'])
        for stage in stages:
            plan.append({'id': f'{name}.{stage}', 'profile': name, 'model': None,
                         'models': candidates, 'config': f'config/{name}.toml',
                         'module': 'sklearn_ml_toolbox.analysis_workflow',
                         'args': [f'--stage={stage}', f'--run-dir=results/{name}']})
    plan.append({**plan[0], 'id': 'report', 'args': ['--stage=report', '--run-dir=results']})
    return plan


def write_task_config(config, step, filename):
    """Materialize a normal CLI config, pointing only to the copied run inputs."""
    settings = task_settings(config, step['profile'], step['models'])
    for key in ('path', 'prepare'):
        value = settings['data'].get(key)
        if value and not (key == 'prepare' and value.startswith('builtin:')):
            path = Path(value)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'data.{key} must be relative to the dataset directory')
            settings['data'][key] = '../' + path.as_posix()
    settings['output'] = {'directory': '../results'}
    text = []
    for section, values in settings.items():
        text.append(f'[{section}]')
        for key, value in values.items():
            text.append(f'{key} = {json.dumps(value, ensure_ascii=False, allow_nan=False)}')
        text.append('')
    filename.write_text('\n'.join(text), encoding='utf-8')
    resolved = load_config(filename)
    if not resolved.input.is_file():
        raise ValueError(f'Input was not copied into the run: {resolved.input}')
    if resolved.prepare and not resolved.prepare.startswith('builtin:') and not Path(resolved.prepare).is_file():
        raise ValueError(f'Preparation hook was not copied into the run: {resolved.prepare}')


def write_overview(directory):
    import pandas as pd

    sections, rows = [], []
    for task in sorted(directory.iterdir()):
        if not task.is_dir() or not (task / 'report.html').exists():
            continue
        manifest = json.loads((task / 'manifest.json').read_text())
        metrics = json.loads((task / 'metrics.json').read_text())
        rows.append({'profile': task.name, 'task': manifest['config']['task'],
                     'target': manifest['config']['target'], 'selected_model': manifest['selected_model'],
                     **{key: value for key, value in metrics.items() if isinstance(value, (int, float)) or value is None}})
        sections.append(f'<li><a href="{task.name}/report.html">{escape(task.name)}: full analysis</a> '
                        f'(<a href="{task.name}/inference/holdout_predictions.csv">saved-model holdout predictions</a>)</li>')
    if not rows:
        raise ValueError('No completed task reports found')
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / 'summary.csv', index=False)
    (directory / 'report.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Analysis workflow</title>'
        '<h1>Analysis workflow</h1><p>Each task has its own training/holdout split. '
        'Model selection uses training cross-validation; metrics from different tasks are not directly comparable.</p>'
        '<ul>' + ''.join(sections) + '</ul>' + summary.to_html(index=False) + '</html>', encoding='utf-8')


def execute_stage(config_path, stage, directory):
    """Commit only complete stages, so a failed report can be retried safely."""
    directory = Path(directory).resolve()
    if stage == 'report':
        write_overview(directory)
        return
    config = load_config(config_path)
    staging = directory.parent.parent / '.staging' / f'{directory.name}_{stage}_{uuid4().hex[:8]}'
    backup = staging.with_name(staging.name + '_previous')
    if stage != 'prepare':
        shutil.copytree(directory, staging)
    try:
        if stage == 'prepare':
            if directory.exists():
                raise FileExistsError(f'Task already prepared: {directory}')
            prepare_run(config, run=staging)
        elif stage == 'train':
            train_run(config, staging)
        elif stage == 'analyze':
            analyze_run(config, staging)
        elif stage == 'evaluate':
            evaluate_run(config, staging)
        elif stage == 'predict':
            predict_csv(staging / 'model.joblib', staging / 'holdout.csv',
                        staging / 'inference/holdout_predictions.csv')
        else:
            raise ValueError(f'Unknown stage: {stage}')
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            directory.rename(backup)
        try:
            staging.rename(directory)
        except OSError:
            if backup.exists():
                backup.rename(directory)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        # The committed task directory remains usable; retain errors in the step log.
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'analyze', 'train', 'evaluate', 'predict', 'report'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    execute_stage(args.config_path, args.stage, args.run_dir)


if __name__ == '__main__':
    main()
