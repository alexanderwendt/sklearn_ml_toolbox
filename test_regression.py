"""Run the grades regression example (not an import-time test runner)."""
from pathlib import Path
from sklearn_ml_toolbox.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['workflow', '--config', str(Path(__file__).parent/'samples/grades/workflow.toml'),
                          '--tasks', 'regression']))
