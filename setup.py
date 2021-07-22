from setuptools import setup, find_packages

setup(
    author='Abhishek Bhatia',
    author_email='bhatiaabhishek8893@gmail.com',
    name='tfutils',
    version='0.0.1',
    packages=find_packages(include=['tfutils', 'tfutils.*']),
    python_requires='>=3.6.*'
)