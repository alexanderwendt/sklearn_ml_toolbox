import subprocess
import sys


def test_package_import_has_no_cli_or_file_side_effects(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import sklearn_ml_toolbox", "--unrelated"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_all_modules_import_without_arguments_or_output_files(tmp_path):
    script = "import importlib,pkgutil,sklearn_ml_toolbox; [importlib.import_module(m.name) for m in pkgutil.walk_packages(sklearn_ml_toolbox.__path__, sklearn_ml_toolbox.__name__+'.')]"
    result = subprocess.run([sys.executable, '-c', script, '--unrelated'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
