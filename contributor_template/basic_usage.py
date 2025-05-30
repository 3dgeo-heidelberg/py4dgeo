"""
===================================================
BASIC USAGE TEMPLATE — PY4DGEO CONTRIBUTION DEMO
===================================================

[MANDATORY METADATA — FILL BEFORE SUBMITTING]

Method Name:       MyMethod (e.g., M3C2, ChangeDetector)
Implemented in:    py4dgeo.changes.my_method
Author(s):            Your Name (Your Institution)
Reference:         Smith, J., & Doe, A. (2023). MyMethod: A new way to detect change.
                   Remote Sensing Journal, 12(3), 123-145. https://doi.org/xxxxx
Required Dataset:  minimal_dataset.npy (provided in `example_data/`) OR public link: https://...
Method description: (e.g. This method detects changes in point clouds using a novel algorithm.)
"""

# [RECOMMENDED] Tested with py4dgeo version X.Y.Z
# Python version >= 3.8


# Imports
from py4dgeo import your_module  # Replace with actual


# ============================================
# Load Example Data
# ============================================
def load_test_data():
    """
    Load or generate minimal working data for testing the method.

    [TODO] Replace this with actual input. Acceptable formats:


    If using public datasets, provide citation and access instructions.
    """
    return None  # replace


# ============================================
# Apply the Method
# ============================================
def apply_method(data):
    """
    Run the core method and return output.

    [TODO] Replace this with a real function call to your method.
    Ensure default parameters are sensible.
    """
    result = "your_result"
    return result


# ============================================
# Run (not mandatory)
# ============================================
if __name__ == "__main__":
    data = load_test_data()
    result = apply_method(data)

    # [TODO] Optional visualization or printout
    print("Output:", result)
