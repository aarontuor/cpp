from setuptools import setup, find_packages

setup(name='safekit',
      version=0.01,
      description='Neural Network Anomaly Detection for Multivariate Sequences',
      url='http://aarontuor.site',
      author='safekit authors',
      author_email='tuora@wwu.edu',
      license='MIT',
      packages=find_packages(), # or list of package paths from this directory
      zip_safe=False,
      install_requires=[],
      classifiers=['Programming Language :: Python'],
      keywords=['Deep Learning', 'Anomaly Detection'])
