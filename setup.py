import subprocess

import setuptools

from codecs import open
from os import path



class BinaryDistribution(setuptools.Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(self):
        return True


subprocess.call(['make', '-C', 'remote_op'])

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()


setuptools.setup(
    name='graphpipe_tf',
    version='1.0.0',
    description='Graphpipe helpers for TensorFlow remote ops',
    long_description=long_description,
    author='OCI ML Team',
    author_email='vish.ishaya@oracle.com',
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
