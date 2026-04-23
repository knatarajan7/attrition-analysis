import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    # Sales: 2 of 3 leave (66.67%), HR: 0 of 2 leave (0%), IT: 1 of 1 leaves (100%)
    # Overtime Yes: all 3 leave (100%), No: none of 3 leave (0%)
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "Sales", "HR", "HR", "IT"],
            "monthly_income": [4000, 5000, 6000, 7000, 8000, 3000],
            "job_satisfaction": [1, 2, 3, 4, 1, 2],
            "overtime": ["Yes", "Yes", "No", "No", "No", "Yes"],
            "attrition": ["Yes", "Yes", "No", "No", "No", "Yes"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_expected_percent(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_zero_when_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_hundred_when_all_leave():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = dict(zip(result["department"], result["attrition_rate"]))
    assert rates["Sales"] == 66.67
    assert rates["HR"] == 0.0
    assert rates["IT"] == 100.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    assert rates["Yes"] == 100.0
    assert rates["No"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    # Leavers: 4000, 5000, 3000 → mean 4000.0
    # Stayers: 6000, 7000, 8000 → mean 7000.0
    result = average_income_by_attrition(sample_df)
    income = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert income["Yes"] == 4000.0
    assert income["No"] == 7000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_within_group_rate(sample_df):
    # Satisfaction 1: 2 employees, 1 leaver → 50% (not 33.33% of all leavers)
    # Satisfaction 2: 2 employees, 2 leavers → 100%
    # Satisfaction 3: 1 employee, 0 leavers → 0%
    # Satisfaction 4: 1 employee, 0 leavers → 0%
    result = satisfaction_summary(sample_df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 50.0
    assert rates[2] == 100.0
    assert rates[3] == 0.0
    assert rates[4] == 0.0


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
