from setuptools import setup, find_packages


setup(
    name="dummy_object_dataset",
    version="0.0.2",
    author="Gabor Vecsei",
    install_requires=['numpy', 'opencv-python'],
    packages=find_packages(),
)
