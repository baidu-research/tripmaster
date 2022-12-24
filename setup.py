#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Setup script.

Authors: sunmingming01(sunmingming01@baidu.com)
Date:    2022/07/12 12:33:34
"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='pangu-framework',
    version='1.1.1-beta',
    description='A High Level Framework for Deep Learning App and Pipelines.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='Apache',
    packages=find_packages(include=["pangu", "pangu.*"]),
    include_package_data=True,
    author='Mingming Sun',
    author_email='sunmingming01@baidu.com',
    keywords=['Deep Learning', 'Framework', 'Pipelines'],
    url='',
    download_url=''
)

install_requires = [
    'addict>=2.4.0',
    'bidict>=0.21.4',
    'more-itertools>=8.12.0',
    'networkx>=2.6.3',
    'port-for>=0.6.1',
    'schema>=0.7.5',
    'tableprint>=0.9.1',
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)