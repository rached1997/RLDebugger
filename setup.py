from distutils.core import setup


def read_requirements(file_name='requirements.txt'):
    with open(file_name) as f:
        return f.read().splitlines()


setup(name='RLDebugger',
      version='0.1',
      description='',
      author='Rached and Ahmed',
      packages=['debugger'],
      install_requires=read_requirements()
      )
