#!/usr/bin/env python3

import os
import yaml


_cfgs_file = os.path.join(os.path.dirname(__file__), "..", "configs.yml")


def load_cfgs():
    res = yaml.load(open(_cfgs_file), Loader=yaml.SafeLoader)
    return res
