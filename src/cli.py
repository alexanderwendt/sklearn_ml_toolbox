"""Portable command-line entry point."""
import argparse
from pathlib import Path
import sys

from .config import load_config
from .workflow import run_pipeline, prepare_run, train_run, evaluate_run
from .prediction.inference import predict_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSV → classification/regression → metrics, predictions and graphs")
    commands = parser.add_subparsers(dest="command", required=True)
    workflow = commands.add_parser('workflow', help='Run a legacy step workflow in a new isolated folder')
    workflow.add_argument('--config', type=Path, required=True)
    workflow.add_argument('--models', help='Comma-separated model names; defaults to all configured models')
    workflow.add_argument('--resume', type=Path, help='Resume this run, skipping completed steps')
    workflow.add_argument('--until', help='Pause after this step ID (see --dry-run)')
    workflow.add_argument('--dry-run', action='store_true', help='Print the expanded steps without creating a run')
    for name in ("run", "prepare", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in ("train", "evaluate"):
            command.add_argument("--run-dir", type=Path, required=True)
    predict = commands.add_parser("predict", help="Load a trusted model and predict prepared CSV features")
    for option in ("model", "input", "output"):
        predict.add_argument(f"--{option}", type=Path, required=True)
    predict.add_argument("--config", type=Path, help="Replay preparation on raw prediction data")
    for name in ("postprocess", "backtest", "analyze-temporal"):
        command = commands.add_parser(name, help="Explicit temporal research utility; input rows must be chronological")
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "postprocess":
            command.add_argument("--threshold", type=float, required=True)
        if name == "backtest":
            command.add_argument("--cost", type=float, default=0.001)
    args = parser.parse_args(argv)
    try:
        if args.command == 'workflow':
            from .workflow_runner import run_workflow
            output = run_workflow(args.config, models=args.models.split(',') if args.models else None,
                                  resume=args.resume, until=args.until, dry_run=args.dry_run)
            if output is None:
                return 0
        elif args.command in ("postprocess", "backtest", "analyze-temporal"):
            import numpy as np
            import pandas as pd
            from .prediction.temporal import backtest
            from .modeling.evaluation import save_plot
            if args.output.exists() or args.output.with_suffix('.png').exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            frame = pd.read_csv(args.input)
            if args.command == "postprocess":
                values = pd.to_numeric(frame['predicted'], errors='raise')
                if not np.isfinite(values).all() or not np.isfinite(args.threshold):
                    raise ValueError("Predictions and threshold must be finite")
                frame['signal'] = (values >= args.threshold).astype(int)
            elif args.command == "backtest":
                frame = backtest(frame, args.cost)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            columns = ['equity', 'buy_hold'] if args.command == 'backtest' else list(frame.select_dtypes(include='number').columns[:8])
            if columns:
                save_plot(args.output.with_suffix('.png'), lambda ax: frame[columns].plot(ax=ax, title=args.command))
            output = args.output
        elif args.command == "run":
            output = run_pipeline(args.config)
        elif args.command == "predict":
            output = predict_csv(args.model, args.input, args.output, load_config(args.config) if args.config else None)
        else:
            config = load_config(args.config)
            output = prepare_run(config) if args.command == "prepare" else {"train": train_run, "evaluate": evaluate_run}[args.command](config, args.run_dir)
        print(output)
        return 0
    except (OSError, ValueError, ImportError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
