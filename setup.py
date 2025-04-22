import os
from setuptools import setup, find_packages
import pathlib
metadata = {
            'tags': {'version': '0.0.0'}, 
            'description': 'Generative AI applied to QAOA parameter optimization'
           }

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

# Get install requires
requirements = f'{os.path.dirname(os.path.realpath(__file__))}/requirements.txt'

setup(
      name = 'ai_quantum',
      version = metadata['tags']['version'],
      description = metadata['description'],
      author = 'Luan Henrique Costa',
      author_email = 'luan.costa@dcc.ufmg.br',
      url = '',
      packages = find_packages(exclude=('notebooks')),
      entry_points = {
                      'console_scripts': [
                                         ],
                     }
     )