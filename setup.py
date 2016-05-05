from setuptools import setup

setup(name='jif',
      version='0.1',
      description='Joint Image Framework for galaxy modeling',
      url='https://github.com/mdschneider/JIF',
      author='Michael D. Schneider, William A. Dawson',
      author_email='mdschneider@me.com',
      license='MIT',
      packages=['jif'],
      package_data={'jif': ['input/*.dat', 'input/*.sed']},
      zip_safe=False)
