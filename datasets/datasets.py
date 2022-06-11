# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the Bongard-HOI library
# which was released under the NVIDIA Source Code Licence.
#
# Source:
# https://github.com/NVlabs/Bongard-HOI
#
# The license for the original version of this file can be
# found in https://github.com/NVlabs/Bongard-HOI/blob/master/LICENSE
# The modifications to this file are subject to the same NVIDIA Source Code Licence.
# ---------------------------------------------------------------


import os


datasets = {}
def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    dataset = datasets[name](**kwargs)
    return dataset
