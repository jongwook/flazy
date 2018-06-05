from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='flazy',
    version='0.0.1',
    description='A sample Python project',
    long_description=long_description,
    url='https://github.com/jongwook/flazy',
    author='Jong Wook Kim',
    author_email='jongwook@nyu.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='dataset functional lazy',
    packages=find_packages(),
    install_requires=['tqdm', 'numpy', 'pandas', 'multiprocess', 'tfrecord_lite'],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['pytest', 'tensorflow==1.8.0'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/jongwook/flazy/issues',
        'Source': 'https://github.com/jongwook/flazy',
    },
)
