from setuptools import setup
import versioneer
from os import path

here = path.abspath(path.dirname(__file__))

# Read the requirements from requirements.txt
requires = open('requirements.txt').read().strip().split('\n')

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

short_description = (
    """Assemble thumbnail-sized images from a large collection into a tiling
    which, viewed at a distance, gives the impression of one large photo.""")

extras_require = {
      'parallel': ['dask'],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='photomosaic',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=short_description,
    long_description=long_description,
    url='https://github.com/danielballan/photomosaic',
    author='Photomosiac Contributors',
    author_email='daniel.b.allan@gmail.com',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='art image color mosaic',
    packages=['photomosaic'],
    install_requires=requires,
    extras_require=extras_require
)
