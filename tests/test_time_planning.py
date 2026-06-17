from datetime import date

from clipopedia.config import Settings
from clipopedia.models import TimeFilter, TimeMode
from clipopedia.retrieval.time_planning import (
    date_range_to_filter,
    make_search_plan,
)

SETTINGS = Settings()


def test_no_constraint_single_bucket():
    plan = make_search_plan(TimeFilter(), SETTINGS)
    assert len(plan) == 1
    assert plan[0].label == "all"


def test_latest_has_recency_first_bucket():
    tf = TimeFilter(has_time_constraint=True, mode=TimeMode.latest)
    plan = make_search_plan(tf, SETTINGS)
    labels = {b.label for b in plan}
    assert "recency_first" in labels
    recency = next(b for b in plan if b.label == "recency_first")
    assert recency.sort_by_date is True


def test_strict_between_is_single_in_range_bucket():
    tf = TimeFilter(
        has_time_constraint=True,
        mode=TimeMode.between,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 1),
    )
    plan = make_search_plan(tf, SETTINGS)
    assert len(plan) == 1
    assert plan[0].label == "in_range"
    assert plan[0].include_date_clause is True


def test_soft_before_adds_backstop():
    tf = TimeFilter(has_time_constraint=True, mode=TimeMode.before, end_date=date(2026, 1, 1))
    plan = make_search_plan(tf, SETTINGS)
    labels = {b.label for b in plan}
    assert labels == {"in_range", "all"}


def test_date_range_to_filter():
    assert date_range_to_filter((20260101, 20260301)) == {
        "pdnumeric": {"$gte": 20260101, "$lte": 20260301}
    }
    assert date_range_to_filter((None, 20260301)) == {"pdnumeric": {"$lte": 20260301}}
    assert date_range_to_filter(None) is None
