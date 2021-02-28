from setuptools import setup
import os


REQUIREMENTS = open("REQUIREMENTS.txt").read().split()
VERSION = open("VERSION.txt").read()
READ_ME = open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read()

# See note below for more information about classifiers
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  "Operating System :: OS Independent",
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3.7',
  'Programming Language :: Python :: 3.8',
  'Programming Language :: Python :: 3.9',
]
 
setup(
  name='pyfuzzyset',
  version=VERSION,
  description='Includes Fuzzy Set Algorithms, Intuitionistic Fuzzy Set Algorithms and Fuzzy Cluster Validity Indexes',
  long_description=READ_ME,
  long_description_content_type='text/markdown',
  url='https://github.com/ibrahimayaz/pyfuzzyset',  # the URL of your package's home page e.g. github link
  author='İbrahim AYAZ, Fatih KUTLU',
  author_email='ayazibrahim13@gmail.com',
  license='MIT License', # note the American spelling
  classifiers=classifiers,
  keywords='', # used when people are searching for a module, keywords separated with a space
  packages='',
  install_requires=REQUIREMENTS,
  python_requires=">=3.7",
)