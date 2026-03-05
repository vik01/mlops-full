"""
TODO: Replace print statements with standard library logging in a later
     session
TODO: Any temporary or hardcoded variable or parameter will be imported from
      config.yml in a later session
"""

from typing import Optional, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder,
)


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    min_max_cols: Optional[List[str]] = None,
    ordinal_encode_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: list of numeric column names to bin into
      quantile buckets
    - categorical_onehot_cols: list of categorical column names to
      one-hot encode
    - numeric_passthrough_cols: list of numeric column names to pass
      through unchanged
    - n_bins: number of quantile bins for KBinsDiscretizer (default 3)
    Outputs:
    - A scikit-learn ColumnTransformer object (NOT yet fitted — no data
      is transformed here)
    Why this contract matters for reliable ML delivery:
    - Returning an unfitted transformer enforces that fitting only happens
      on training data, preventing data leakage from the test set. The same
      recipe object is passed into train.py, fitted there, and bundled into
      the saved Pipeline so that inference automatically applies the identical
      transformations.
    """

    # TODO: replace with logging later
    print("[features] Building feature preprocessor recipe...")

    # Apply defaults: treat None as empty list
    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    min_max_cols = min_max_cols or []
    ordinal_encode_cols = ordinal_encode_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # Build the list of (name, transformer, columns) tuples.
    # A transformer step is only added if its column list is non-empty
    # so the pipeline stays clean when a list is not used.
    transformers = []

    # ------------------------------------------------------------------
    # TRANSFORMER 1: Quantile binning for numeric columns
    # KBinsDiscretizer splits a continuous numeric column into n
    # equal-frequency (quantile) buckets and outputs an integer
    # bucket index.
    # Example: Age [22, 35, 55, 70] with n_bins=3  ->  [0, 0, 1, 2]
    # encode="ordinal" keeps output as integers (not one-hot sparse),
    # which is efficient and works well with linear and tree-based
    # ------------------------------------------------------------------
    if quantile_bin_cols:
        bin_transformer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile",
        )
        transformers.append(
            ("quantile_bin", bin_transformer, quantile_bin_cols)
        )

    # ------------------------------------------------------------------
    # TRANSFORMER 2: One-hot encoding for categorical columns
    # ------------------------------------------------------------------
    if categorical_onehot_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("onehot", ohe, categorical_onehot_cols))

    # ------------------------------------------------------------------
    # TRANSFORMER 3: Passthrough for numeric columns that don't need
    # transform
    # ------------------------------------------------------------------
    if numeric_passthrough_cols:
        transformers.append(
            ("passthrough", "passthrough", numeric_passthrough_cols)
        )

    # ------------------------------------------------------------------
    # TRANSFORMER 3: MinMaxScaler for numeric columns that need
    # zero-mean normalisation
    # ------------------------------------------------------------------
    if min_max_cols:
        transformers.append(("scaler", MinMaxScaler(), min_max_cols))

    # ------------------------------------------------------------------
    # TRANSFORMER 4: OrdinalEncoder for high-cardinality categorical
    # columns
    # ------------------------------------------------------------------
    if ordinal_encode_cols:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        transformers.append(("ordinal", ordinal_encoder, ordinal_encode_cols))

    # ------------------------------------------------------------------
    # Assemble the ColumnTransformer
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor
