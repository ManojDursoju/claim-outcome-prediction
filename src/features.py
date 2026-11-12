
import pandas as pd


def calculate_training_rates(df, column):
    """Calculate leave-one-out denial rates for training rows."""

    denied = (df["status"] == "Denied").astype(int)

    grouped_denials = denied.groupby(df[column]).transform("sum")
    grouped_counts = df.groupby(column)[column].transform("count")

    # Exclude the current row to avoid target leakage
    rates = (grouped_denials - denied) / (grouped_counts - 1).clip(lower=1)

    return rates


def calculate_reference_rates(df, column):
    """Calculate category denial rates using training data only."""

    return (
        df.assign(is_denied=(df["status"] == "Denied").astype(int))
        .groupby(column)["is_denied"]
        .mean()
        .to_dict()
    )


def add_training_features(df):
    """Add leakage-safe features to training data."""

    result = df.copy()

    result["payer_denial_rate"] = calculate_training_rates(
        result, "payer"
    )

    result["provider_denial_rate"] = calculate_training_rates(
        result, "provider"
    )

    return result


def add_reference_features(df, payer_rates, provider_rates, global_rate):
    """Add features to test or inference data using training references."""

    result = df.copy()

    result["payer_denial_rate"] = (
        result["payer"].map(payer_rates).fillna(global_rate)
    )

    result["provider_denial_rate"] = (
        result["provider"].map(provider_rates).fillna(global_rate)
    )

    return result
