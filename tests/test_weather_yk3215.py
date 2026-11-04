import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from weather_yk3215.weather_yk3215 import (
    celsius_to_fahrenheit,
    summarize_temperatures,
    clean_readings,
)

@pytest.mark.parametrize("celsius, expected", [
    (0, 32),
    (100, 212),
    (-40, -40),
    (37, 98.6),
    (-273.15, -459.67)
])
def test_celsius_to_fahrenheit(celsius, expected):
    """Test Celsius to Fahrenheit conversion."""
    result = celsius_to_fahrenheit(celsius)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("bad_input, expected_error", [
    ("hello", TypeError),
    (None, TypeError),
    (True, TypeError),
    (-300, ValueError)
])
def test_celsius_to_fahrenheit_invalid(bad_input, expected_error):
    """Test that invalid inputs raise the expected errors."""
    with pytest.raises(expected_error):
        celsius_to_fahrenheit(bad_input)


def test_summarize_temperatures():
    """Test temperature statistics with missing values."""
    # Arrange
    temps = [10.0, float("nan"), 20.0]

    # Act
    result = summarize_temperatures(temps)

    # Assert
    assert result["mean"] == pytest.approx(15.0)
    assert result["min"] == 10.0
    assert result["max"] == 20.0
    assert result["n_missing"] == 1


@pytest.fixture
def raw_readings():
    """Create sample weather readings for testing."""
    return pd.DataFrame({
        "station": ["JFK", "CPK", "JFK", "CPK", "LGA"],
        "date": ["2026-07-02", "2026-07-01", "2026-07-01",
                 "2026-07-02", "2026-07-01"],
        "temp_c": [30.0, 25.0, float("nan"), 999.0, 28.0],
    })


def test_raw_readings_fixture(raw_readings):
    """Test the fixture's columns, rows, and missing values."""
    assert list(raw_readings.columns) == ["station", "date", "temp_c"]
    assert len(raw_readings) == 5
    assert raw_readings["temp_c"].isna().sum() == 1


def test_clean_readings(raw_readings):
    """Test that raw readings are cleaned correctly."""
    # Act
    result = clean_readings(raw_readings)

    # Arrange expected result
    expected = pd.DataFrame({
        "station": ["CPK", "JFK", "LGA"],
        "date": ["2026-07-01", "2026-07-02", "2026-07-01"],
        "temp_c": [25.0, 30.0, 28.0],
        "temp_f": [77.0, 86.0, 82.4],
    })

    # Assert
    assert_frame_equal(result, expected)

