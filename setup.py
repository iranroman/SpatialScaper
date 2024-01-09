from setuptools import setup
import imp


with open('README.md') as file:
    long_description = file.read()

version = imp.load_source('spatial_scaper.version', 'spatial_scaper/version.py')

setup(
    name='spatial_scaper',
    version=version.version,
    description='A library for soundscape synthesis and augmentation using spatial impulse responses inside specific rooms',
    author='Iran R. Roman',
    author_email='iran@ccrma.stanford.edu',
    url='',
    download_url='http://github.com/iranroman/SELD-data-generator/releases',
    packages=['spatial_scaper'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='audio sound soundscape environmental ambisonics microphone array sound event detection localization',
    license='Creative Commons Attribution',
    classifiers=[
            "License :: Creative Commons Attribution 4.0",
            "Programming Language :: Python",
            "Development Status :: 3 - Alpha",

            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Multimedia :: Sound/Audio :: Analysis",
            "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        ],
    install_requires=[
        'numpy>=1.24.4',
        'scipy>=1.11.1',
        'librosa>=0.10.0',
        'mat73>=0.60',
        'fvcore>=0.1.5',
        'pysofaconventions',
        'pyroomacoustics',
    ],

)
