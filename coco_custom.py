from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    METAINFO = {
        "classes": (
            "ferritin_complex",
            "beta_amylase",
            "beta_galactosidase",
            "cytosolic_ribosome",
            "thyroglobulin",
            "virus",
        ),
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
        ],
    }
