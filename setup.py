from setuptools import setup

setup(name='jif',
      version='1.0',
      description='Joint Image Framework for galaxy modeling',
      url='https://github.com/mdschneider/JIF',
      author='Michael D. Schneider, William A. Dawson',
      author_email='schneider42@llnl.gov',
      license='GNU',
      packages=['jif', 'jiffy'],
      package_data={'jif': ['input/*.dat', 'input/*.sed']},
      ### The python scripts in this package can be designated as distinct command-line executables
      ### here. Be careful about potential name clashes.
      entry_points = {
        'console_scripts': ['jif_sheller=jif.sheller:main',
                            'jif_roaster=jif.Roaster:main',
                            'jif_roaster_inspector=jif.RoasterInspector:main',
                            'jiffy_roaster=jiffy.roaster:main',
                            'jiffy_roaster_inspector=jiffy.roaster_inspector:main',
                            'jiffy_stooker=jiffy.stooker:main'],
      },
      zip_safe=False)
