from setuptools import setup

setup(name='phtomosiac',
      version='0.1.0',
      author='Daniel Allan and David Lu',
      packages=['photomosaic'],
      description='A persistent dictionary with history backed by sqlite',
      url='http://github.com/danielballan/photomosaic',
      platforms='Cross platform (Linux, Mac OSX, Windows)',
      requires=['sqlite', 'PIL']
      )
