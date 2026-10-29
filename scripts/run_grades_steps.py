"""Run historical grades diagnostics against an existing classification run.

Uses only its training split for exploratory selection. Each subprocess has a
separate log; unavailable historical stages are reported, never silently passed.
"""
import argparse
import configparser
from html import escape
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from urllib.parse import quote

import joblib
import pandas as pd
from sklearn.base import clone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    out = run / 'historical_steps'
    out.mkdir(exist_ok=False)
    data = out / 'prepared'
    data.mkdir()
    logs = out / 'logs'
    logs.mkdir()
    parts = joblib.load(run / 'split.joblib')
    model = joblib.load(run / 'model.joblib')
    train = parts['train']
    X = train[model['features']].apply(pd.to_numeric)
    y = train[[model['target']]].astype(int)
    for name, frame in [('features_raw', X), ('outcomes_raw', y), ('source', X.join(y))]:
        frame.to_csv(data / f'{name}.csv', sep=';', index_label='id')
    (data / 'labels.csv').write_text(''.join(f'Grade {label};{label}\n' for label in sorted(y.iloc[:, 0].unique())))
    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read(root / 'samples/grades/config/grades.ini')
    conf['Common'].update(class_name='GradeClass', problem_type='classification')
    conf['Paths'].update(prepared_data_directory=data.as_posix(), results_directory=out.as_posix(),
                         models_directory=(out / 'models').as_posix(), config_directory=out.as_posix(),
                         labels_path=(data / 'labels.csv').as_posix())
    conf['Preparation'].update(source_in=(data / 'source.csv').as_posix(), test_size='0.2', shuffle_data='True')
    # Legacy split is an inner split of training data, not the untouched holdout.
    conf['Training'].update(features_val_in='${Preparation:features_out_val}',
                            outcomes_val_in='${Preparation:outcomes_out_val}')
    ini = out / 'grades.ini'

    def save_config():
        with ini.open('w') as stream:
            conf.write(stream)

    records = []

    def execute(module, *extra, label=None, timeout=600):
        name = label or module.rsplit('.', 1)[-1]
        snapshot = out / f'{name}.ini'
        with snapshot.open('w') as stream:
            conf.write(stream)
        command = [sys.executable, '-m', f'sklearn_ml_toolbox.{module}', '--config_path', str(snapshot), *extra]
        print(f'Running {name}', flush=True)
        with (logs / f'{name}.log').open('w', encoding='utf-8') as stream:
            try:
                result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONHASHSEED': '42'}, timeout=timeout)
                status = 'completed' if result.returncode == 0 else f'failed (exit {result.returncode})'
            except subprocess.TimeoutExpired:
                status = f'timed out after {timeout}s'
        records.append({'stage': name, 'status': status, 'command': subprocess.list2cmdline(command)})
        pd.DataFrame(records).to_csv(out / 'stage_status.csv', index=False)

    save_config()
    execute('data_processing.step30_clean_raw_data')
    execute('data_processing.step31_adapt_features')
    execute('data_processing.step36_split_training_validation')
    execute('data_processing.step35_perform_feature_selection')
    execute('data_processing.step33_analyze_data')
    execute('data_processing.step32_search_hyperparameters', timeout=900)
    # Exercise historical search entry points as requested; logs capture legacy
    # compatibility failures. The shared run above supplies all-model CV results.
    execute('modeling.step42_analyze_training_time', timeout=180)
    execute('modeling.step43_wide_hyperparameter_search_all', '--debug_param', timeout=180)
    execute('modeling.step44_narrow_hyperparameter_search_all', timeout=180)
    # Run the historical final-training stage for every selected modern pipeline.
    for name, pipeline in model['candidates'].items():
        directory = out / 'models' / name
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'pipe.pkl').open('wb') as stream:
            pickle.dump(clone(pipeline), stream)
        conf['ModelTrain'].update(features_in='${Preparation:features_out}', outcomes_in='${Preparation:outcomes_out}',
                                  pipeline_in=(directory / 'pipe.pkl').as_posix(),
                                  model_out=(directory / 'model.joblib').as_posix())
        (directory / 'ext_param.json').write_text(json.dumps({'pr_threshold': 0.5}))
        conf['ModelTrain']['ext_param_in'] = (directory / 'ext_param.json').as_posix()
        conf['Paths']['results_directory'] = directory.as_posix()
        save_config()
        execute('modeling.step50_train_model_from_pipe', '--config_section', 'ModelTrain', label=f'step50_{name}')
        execute('modeling.step60_evaluate_model', '--config_section', 'EvaluationValidation', label=f'step60_{name}', timeout=180)
    records.extend([
        {'stage': 'step20/21/22/34/61/70/71/72', 'status': 'not applicable: market prices, dates, annotations or trading signals required'},
        {'stage': 'step45', 'status': 'not applicable: binary threshold tuning; GradeClass has five classes'},
        {'stage': 'scripts/download_samples.py, migrate_samples.py, run_samples.py', 'status': 'dataset acquisition/migration/multi-dataset orchestration; grades already local'},
        {'stage': 'shared preparation/search/training/evaluation/inference', 'status': 'see parent report and model artifacts'},
    ])
    table = pd.DataFrame(records)
    table.to_csv(out / 'stage_status.csv', index=False)
    selections = data / 'selected_feature_columns.csv'
    selected_html = pd.read_csv(selections, sep=';').fillna('').to_html(index=False) if selections.exists() else '<p>See step35 log.</p>'
    images = ''.join(f'<figure><img loading="lazy" src="{quote(p.relative_to(out).as_posix())}"><figcaption>{escape(p.stem)}</figcaption></figure>' for p in sorted(out.rglob('*.png')))
    links = ''.join(f'<li><a href="{quote(p.relative_to(out).as_posix())}">{escape(p.relative_to(out).as_posix())}</a></li>' for p in sorted(out.rglob('*')) if p.suffix in ('.csv', '.log', '.ini'))
    (out / 'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Grades historical stages</title>
<style>body{{font:16px sans-serif;margin:2em}}img{{max-width:100%;max-height:1000px}}td,th{{padding:.4em}}table{{border-collapse:collapse}}</style>
<h1>Grades classification: historical stages</h1><p><a href="../report.html">All-model comparison</a></p>
<p>Exploratory diagnostics use training rows only. Step36 creates an inner training split. Historical Lasso and OLS backward elimination treat grade codes as numeric: these are ordinal exploratory rankings, not validated multiclass selectors. These feature sets do not influence the main model comparison; its preprocessing and search are fitted inside CV folds.</p>
<h2>Execution status</h2>{table.fillna('').to_html(index=False)}<h2>Step35 selected features</h2>{selected_html}
<h2>Plots</h2>{images}<h2>Logs and tables</h2><ul>{links}</ul></html>''', encoding='utf-8')
    print(out / 'report.html', flush=True)


if __name__ == '__main__':
    main()
