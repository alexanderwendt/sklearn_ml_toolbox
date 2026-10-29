import pandas as pd
import pytest


def test_backtest_signal_applied_next_bar():
    from sklearn_ml_toolbox.prediction.temporal import backtest
    frame = pd.DataFrame({'Close': [100., 200., 100.], 'signal': [1, 0, 1]})
    result = backtest(frame, cost=0)
    assert result.position.tolist() == [0, 1, 0]
    assert result.equity.tolist() == [1, 2, 2]


def test_backtest_charges_turnover_and_rejects_bad_prices():
    from sklearn_ml_toolbox.prediction.temporal import backtest
    frame = pd.DataFrame({'Close': [100., 100., 100.], 'signal': [1, 0, 1]})
    result = backtest(frame, cost=.01)
    assert result.equity.iloc[-1] == pytest.approx(.9801)
    frame.loc[1, 'Close'] = 0
    with pytest.raises(ValueError, match='positive'):
        backtest(frame)


def test_temporal_cli_outputs(tmp_path):
    from sklearn_ml_toolbox.cli import main
    source = tmp_path/'signals.csv'
    pd.DataFrame({'Close': [100, 101, 99, 102], 'predicted': [.9, .1, .8, .5]}).to_csv(source, index=False)
    processed = tmp_path/'processed.csv'
    assert main(['postprocess', '--input', str(source), '--output', str(processed), '--threshold', '.6']) == 0
    assert pd.read_csv(processed).signal.tolist() == [1, 0, 1, 0]
    output = tmp_path/'backtest.csv'
    assert main(['backtest', '--input', str(processed), '--output', str(output)]) == 0
    assert len(pd.read_csv(output)) == 4
    assert output.with_suffix('.png').is_file()
