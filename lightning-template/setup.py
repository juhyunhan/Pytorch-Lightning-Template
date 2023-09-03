from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='Test',
    version=__version__,
    author='SungJun Shin',
    author_email='ssjun511@kau.kr',
    url='',
    license='',
    packages=find_packages(include=['seg_det', 'seg_det.*']),
    zip_safe=False,
)