from py4dgeo.m3c2 import M3C2


_cloudcompare_param_mapping = {
    "normalscale": "normal_radii",
    "registrationerror": "reg_error",
    "searchdepth": "max_distance",
    "searchscale": "cyl_radii",
    "usemedian": "robust_aggr",
}


class CloudCompareM3C2(M3C2):
    def __init__(self, **params):
        """An M3C2 implementation that uses parameter names from CloudCompare"""
        # Remap parameters using above mapping
        py4dgeo_params = {
            _cloudcompare_param_mapping.get(k, k): v for k, v in params.items()
        }

        # Intialize base class with remapped parameters
        super().__init__(**py4dgeo_params)
