"""Configuration-driven adapter for the numbered legacy workflow steps."""
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json
import os
import shutil
import subprocess
import sys
import tomllib


# Define each command once; model loops and configuration paths are expanded later.
PREPARE = [
    ('groundtruth', 'data_generation.step20_generate_groundtruth_stockmarket', []),
    ('features', 'data_generation.step21_generate_features', ['-debug']),
    ('dimensions', 'data_generation.step22_adapt_dimensions', []),
    ('clean', 'data_processing.step30_clean_raw_data', []),
    ('adapt', 'data_processing.step31_adapt_features', []),
    ('analyze', 'data_processing.step33_analyze_data', []),
    ('temporal', 'data_processing.step34_analyze_temporal_data', []),
    ('select', 'data_processing.step35_perform_feature_selection', []),
    ('split', 'data_processing.step36_split_training_validation', []),
]
TRAIN = [
    ('timing', 'modeling.step42_analyze_training_time', []),
    ('wide', 'modeling.step43_wide_hyperparameter_search_all', ['--execute_wide=True', '-debug']),
    ('narrow', 'modeling.step44_narrow_hyperparameter_search_all', []),
    ('threshold', 'modeling.step45_define_precision_recall', []),
    ('fit_train', 'modeling.step50_train_model_from_pipe', ['--config_section=ModelTrain']),
    ('fit_final', 'modeling.step50_train_model_from_pipe', ['--config_section=ModelFinal']),
    ('evaluate_train', 'modeling.step60_evaluate_model', ['--config_section=EvaluationTraining']),
    ('plot_train', 'modeling.step61_evaluate_model_temporal_data', ['--config_section=EvaluationTraining']),
    ('evaluate_validation', 'modeling.step60_evaluate_model', ['--config_section=EvaluationValidation']),
    ('plot_validation', 'modeling.step61_evaluate_model_temporal_data', ['--config_section=EvaluationValidation']),
    ('postprocess', 'prediction.step71_value_postprocessing', ['--config_section=EvaluationValidation']),
    ('backtest', 'prediction.step72_backtesting', ['--config_section=EvaluationValidation']),
]
INFER_PREPARE = [
    PREPARE[1], PREPARE[2],
    ('clean', 'data_processing.step30_clean_raw_data', ['--no_images', '--on_inference_data']),
    PREPARE[4],
]
PREDICT = [('predict', 'prediction.step70_predict_temporal_data', ['--config_section=EvaluationInference'])]


def read_workflow(path):
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    unknown = config.keys() - {'workflow', 'defaults', 'profiles', 'models', 'steps'}
    if unknown:
        raise ValueError(f'Unknown workflow sections: {sorted(unknown)}')
    if config['workflow']['version'] != 1:
        raise ValueError('Unsupported workflow version')
    for profile in ('prepare', 'train', 'infer'):
        if profile not in config['profiles']:
            raise ValueError(f'Missing profile: {profile}')
    if not config.get('models'):
        raise ValueError('At least one model is required')
    for model in config['models']:
        if not model.replace('_', '').isalnum():
            raise ValueError(f'Invalid model name: {model}')
    return config


def resolved_config(config, profile, model=None):
    """Merge defaults < profile < model, retaining legacy INI interpolation."""
    result = ConfigParser(interpolation=ExtendedInterpolation())
    result.read_dict(config.get('defaults', {}))
    result.read_dict(config['profiles'][profile])
    if model:
        result.set('Common', 'model_type', model)
        # Model overrides apply to training; inference shares only the model identity.
        if profile == 'train':
            result.read_dict(config['models'][model])
    # Resolve now so invalid references fail before expensive work begins.
    for section in result.sections():
        for key, value in result.items(section):
            if value and key.endswith(('_path', '_directory', '_in', '_out')):
                artifact = Path(value)
                if artifact.is_absolute() or '..' in artifact.parts:
                    raise ValueError(f'{section}.{key} must be relative to the isolated workspace')
    return result


def build_plan(config, models=None):
    models = list(models or config['models'])
    if len(set(models)) != len(models) or set(models) - config['models'].keys():
        raise ValueError('Models must be unique names declared in workflow.toml')
    plan = []

    def append(group, profile, model, steps):
        filename = f'{profile}_{model}.ini' if model else f'{profile}.ini'
        for name, module, default_args in steps:
            step_id = f'{group}.{name}'
            options = config.get('steps', {}).get(step_id, {})
            if options.keys() - {'args'}:
                raise ValueError(f'Unknown options for {step_id}')
            arguments = options.get('args', default_args)
            if not isinstance(arguments, list) or not all(isinstance(arg, str) for arg in arguments):
                raise ValueError(f'{step_id}.args must be a list of strings')
            if any(arg.startswith(('--config_path', '-conf')) for arg in arguments):
                raise ValueError('Step arguments cannot redirect generated configurations')
            plan.append({'id': step_id, 'profile': profile, 'model': model,
                         'config': f'config/{filename}',
                         'module': f'sklearn_ml_toolbox.{module}', 'args': arguments})

    append('prepare', 'prepare', None, PREPARE)
    for model in models:
        append(model, 'train', model, TRAIN)
    append('infer_prepare', 'infer', models[0], INFER_PREPARE)
    for model in models:
        append(f'{model}.inference', 'infer', model, PREDICT)
    known = {step['id'] for step in build_plan(config_without_steps(config))} if config.get('steps') else set()
    if config.get('steps', {}).keys() - known:
        raise ValueError('Unknown step override; use --dry-run to list step IDs')
    return plan


def config_without_steps(config):
    return {key: value for key, value in config.items() if key != 'steps'}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(run, manifest):
    temporary = run / 'manifest.tmp'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.replace(run / 'manifest.json')


def create_run(path, config, plan):
    output = (path.parent / config['workflow'].get('output', '_old_reruns')).resolve()
    run = output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    workspace = run / 'workspace'
    (workspace / 'config').mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    shutil.copy2(path, run / 'workflow.toml')
    for relative in config['workflow']['inputs']:
        source = (path.parent / relative).resolve()
        target = (workspace / relative).resolve()
        if not source.is_relative_to(path.parent) or not target.is_relative_to(workspace):
            raise ValueError('Input paths must stay within the dataset directory')
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for step in plan:
        filename = workspace / step['config']
        if not filename.exists():
            with filename.open('w', encoding='utf-8') as stream:
                resolved_config(config, step['profile'], step['model']).write(stream)
    # The legacy search definitions are repository-level Python modules.
    repository = Path(__file__).resolve().parent.parent
    shutil.copytree(repository / 'models', run / 'runtime/models', ignore=shutil.ignore_patterns('__pycache__'))
    immutable = [run / 'workflow.toml', *workspace.joinpath('config').glob('*.ini'),
                 *run.joinpath('runtime').rglob('*.py')]
    for relative in config['workflow']['inputs']:
        item = workspace / relative
        immutable.extend(p for p in item.rglob('*') if p.is_file()) if item.is_dir() else immutable.append(item)
    manifest = {'version': 1, 'status': 'created', 'python': sys.version,
                'config_sha256': digest(path), 'plan': plan, 'completed': [], 'attempts': [],
                'snapshots': {p.relative_to(run).as_posix(): digest(p) for p in immutable}}
    write_manifest(run, manifest)
    with (run / 'environment.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    return run, manifest


def run_workflow(path, models=None, resume=None, until=None, dry_run=False):
    path = Path(path).resolve()
    config = read_workflow(path)
    if resume:
        run = Path(resume).resolve()
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['config_sha256'] != digest(path):
            raise ValueError('Workflow config changed; start a new run')
        if models:
            raise ValueError('Resume uses the original model selection; omit --models')
        for relative, expected in manifest['snapshots'].items():
            if digest(run / relative) != expected:
                raise ValueError(f'Run snapshot changed: {relative}; start a new run')
        plan = manifest['plan']
    else:
        plan = build_plan(config, models)
    if until:
        matches = [i for i, step in enumerate(plan) if step['id'] == until]
        if not matches:
            raise ValueError(f'Unknown --until step: {until}')
        selected = plan[:matches[0] + 1]
    else:
        selected = plan
    if dry_run:
        for step in selected:
            print(f"{step['id']}: {step['module']} --config_path={step['config']} {' '.join(step['args'])}")
        return None
    if not resume:
        run, manifest = create_run(path, config, plan)
    print(f'Run directory: {run}', flush=True)
    source = Path(__file__).resolve().parent
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(run / 'runtime'), str(source), env.get('PYTHONPATH')]))
    for step in selected:
        if step['id'] in manifest['completed']:
            continue
        log = f"logs/{len(manifest['attempts']) + 1:03d}_{step['id']}.log"
        command = [sys.executable, '-m', step['module'], f"--config_path={step['config']}", *step['args']]
        attempt = {'step': step['id'], 'command': command, 'log': log,
                   'started': datetime.now(timezone.utc).isoformat(), 'exit_code': None}
        manifest['attempts'].append(attempt)
        manifest.update(status='running', current_step=step['id'])
        write_manifest(run, manifest)
        print(f"[{plan.index(step) + 1}/{len(plan)}] {step['id']}", flush=True)
        with (run / log).open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=run / 'workspace', env=env, stdout=stream, stderr=subprocess.STDOUT)
        attempt.update(exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())
        if result.returncode:
            manifest['status'] = 'failed'
            write_manifest(run, manifest)
            raise ValueError(f"Step {step['id']} failed ({result.returncode}); see {run / log}")
        manifest['completed'].append(step['id'])
        write_manifest(run, manifest)
    manifest.update(status='complete' if len(manifest['completed']) == len(plan) else 'paused', current_step=None)
    write_manifest(run, manifest)
    return run
