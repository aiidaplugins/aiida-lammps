#!/usr/bin/env python

from __future__ import absolute_import

import json

from setuptools import find_packages, setup

if __name__ == "__main__":
    with open("setup.json", "r") as info:
        kwargs = json.load(info)

    setup(
        packages=find_packages(),
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        **kwargs
    )
