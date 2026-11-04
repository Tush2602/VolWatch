from models.splitter import Timesplitter
import pandas as pd

def make_table():
    idx = pd.bdate_range("2015-08-01", "2022-12-31")
    table = pd.DataFrame({"vol":1.0 , "target":1.0}, index= idx)
    table.iloc[-1, table.columns.get_loc("target")]
    return table 

def test_segments_are_chronological_and_disjoint():
    s = Timesplitter("2017-12-31", "2019-12-31").split(make_table())
    assert s.train.index.max() < s.valid.index.min()
    assert s.valid.index.max() < s.test.index.min()


def test_purge_leaves_a_gap_at_each_boundary():
    table = make_table()
    s = Timesplitter("2017-12-31", "2019-12-31").split(table)
    labeled = table.dropna(subset=["target"])
    gap = labeled.index.get_loc(s.valid.index.min()) - labeled.index.get_loc(s.train.index.max())
    assert gap >= 2  # at least one purged row between train and validation


def test_unlabeled_row_is_never_evaluated():
    s = Timesplitter().split(make_table())
    for part in (s.train, s.valid, s.test):
        assert part["target"].notna().all()
    
def test_purge_also_applies_between_validation_and_test():
    table = make_table()
    s = Timesplitter("2017-12-31", "2019-12-31").split(table)
    labeled = table.dropna(subset=["target"])
    gap = labeled.index.get_loc(s.test.index.min()) - labeled.index.get_loc(s.valid.index.max())
    assert gap >= 2