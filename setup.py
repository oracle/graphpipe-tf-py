import subprocess

import setuptools

from codecs import open
from os import path



# FIXME(vish): disabled remote op until it is converted
# class BinaryDistribution(Distribution):
#     """Distribution which always forces a binary package with platform name"""
#     def has_ext_modules(self):
#         return True
#
#
subprocess.call(['make', '-C', 'remote_op'])

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()


setuptools.setup(
    name='graphpipe_tf',
    version='0.0.1',
    description='Graphpipe helpers for TensorFlow remote ops',
    long_description=long_description,
    author='OCI ML Team',
    author_email='internal@oracle.com',
    classifier=[
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    #packages=setuptools.find_packages(exclude=['contrib', 'docs', 'remote_op', 'tests']),
    packages=['graphpipe_tf'],
    package_data={'graphpipe_tf': ['remote_op.so']},

#    distclass=BinaryDistribution,
)
