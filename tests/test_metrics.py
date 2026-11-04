import pytest

from models.metrics import Metrics

def test_rmse_and_mae():
    a,f = [1,2,3],[1, 2, 5]
    assert Metrics.rmse(a,f)== pytest.approx(1.1547, abs=1e-4)
    assert Metrics.mae(a,f) == pytest.approx(2/3)

def test_qlike_is_zero_for_a_perfect_forecast():
    assert Metrics.qlike([5.0, 10.0], [5.0, 10.0]) == pytest.approx(0.0)

def test_qlike_known_values():
    assert Metrics.qlike([2.0], [1.0])== pytest.approx(1.6137, abs=1e-4)

def test_qlike_penalizes_under_prediciton_more():
    assert Metrics.qlike([10.0], [5.0]) > Metrics.qlike([10.0], [20.0])