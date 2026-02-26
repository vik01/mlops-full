"""
Educational Goal:
- Why this module exists in an MLOps system: Feature engineering transformations must be
  defined ONCE and bundled with the model so that training and inference always apply
  the exact same transformations. Without this, you get silent inconsistencies between
  what the model was trained on and what it receives at prediction time — a common cause
  of production bugs.
- Responsibility (separation of concerns): features.py owns ONLY the transformation
  recipe (the "what to do" and "how to do it"). It does NOT load data, fit on data, or
  make predictions. It returns a ColumnTransformer that is later fitted inside train.py.
- Pipeline contract (inputs and outputs):
    Inputs : configuration lists of column names and binning parameters
    Outputs: a scikit-learn ColumnTransformer object (unfitted recipe only)

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: list of numeric column names to bin into quantile buckets
    - categorical_onehot_cols: list of categorical column names to one-hot encode
    - numeric_passthrough_cols: list of numeric column names to pass through unchanged
    - n_bins: number of quantile bins for KBinsDiscretizer (default 3)
    Outputs:
    - A scikit-learn ColumnTransformer object (NOT yet fitted — no data is transformed here)
    Why this contract matters for reliable ML delivery:
    - Returning an unfitted transformer enforces that fitting only happens on training
      data, preventing data leakage from the test set. The same recipe object is passed
      into train.py, fitted there, and bundled into the saved Pipeline so that inference
      automatically applies the identical transformations.
    """

    print("[features] Building feature preprocessor recipe...")  # TODO: replace with logging later
    
    # Apply defaults: treat None as empty list
    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # ------------------------------------------------------------------
    # TRANSFORMER 1: Quantile binning for numeric columns
    # KBinsDiscretizer splits a continuous numeric column into n equal-
    # frequency (quantile) buckets and outputs an integer bucket index.
    # Example: Age [22, 35, 55, 70] with n_bins=3  ->  [0, 0, 1, 2]
    # encode="ordinal" keeps output as integers (not one-hot sparse),
    # which is efficient and works well with linear and tree-based models.
    # ------------------------------------------------------------------
    bin_transformer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
    )

    # ------------------------------------------------------------------
    # TRANSFORMER 2: One-hot encoding for categorical columns
    # OneHotEncoder converts a categorical column (e.g. "color": red/blue)
    # into binary indicator columns (color_red=1/0, color_blue=1/0).
    # handle_unknown="ignore" makes unseen categories at inference time
    # produce an all-zeros row instead of crashing.
    # try/except handles a scikit-learn API change:
    #   sparse_output=False  (scikit-learn >= 1.2)
    #   sparse=False         (scikit-learn <= 1.1)
    # ------------------------------------------------------------------
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    # ------------------------------------------------------------------
    # TRANSFORMER 3: Numeric passthrough
    # "passthrough" is a built-in sklearn keyword — it tells the
    # ColumnTransformer to keep these columns exactly as they are,
    # with no transformation applied.
    # ------------------------------------------------------------------

    # Build the list of (name, transformer, columns) tuples.
    # A transformer step is only added if its column list is non-empty
    # so the pipeline stays clean when a list is not used.
    transformers = []

    if quantile_bin_cols:
        transformers.append(("quantile_bin", bin_transformer, quantile_bin_cols))

    if categorical_onehot_cols:
        transformers.append(("onehot", ohe, categorical_onehot_cols))

    if numeric_passthrough_cols:
        transformers.append(("passthrough", "passthrough", numeric_passthrough_cols))
    
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add or adjust transformers for your specific dataset.
    # Why: Different datasets require different transformation strategies.
    #      The three transformers above cover the most common cases, but
    #      your data may need additional steps.
    # Examples:
    # 1. Add StandardScaler for columns that need zero-mean normalisation:
    #    from sklearn.preprocessing import StandardScaler
    #    transformers.append(("scaler", StandardScaler(), ["income", "age"]))
    # 2. Use OrdinalEncoder instead of OHE for high-cardinality categoricals:
    #    from sklearn.preprocessing import OrdinalEncoder
    #    transformers.append(("ordinal", OrdinalEncoder(), ["education_level"]))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # ------------------------------------------------------------------
    # Assemble the ColumnTransformer
    # remainder="drop" means any column NOT explicitly listed above will
    # be silently dropped. This is intentional — the pipeline only
    # processes columns you have configured in SETTINGS.
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor