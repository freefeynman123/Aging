from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='Aging',
    version='0.1',
    author='Lukasz Bala',
    description='Conditional adversarial autoencoders for face aging',
    long_description=open('README.md').read(),
    url='https://github.com/freefeynman123/Aging',
    install_requires=requirements,
    license='MIT',
    packages=['aging'],
    zip_safe=True
)