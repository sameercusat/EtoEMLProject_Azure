from setuptools import setup,find_packages
from typing import List
HYPHEN_E_DOT='-e .'

def get_requirements(file:str)->List[str]:
    with open(file,'r') as f:
        l=f.readlines()
        l=[i.replace('\n','') for i in l]
        if HYPHEN_E_DOT in l:
            l.remove(HYPHEN_E_DOT)
        return l


setup (name = 'MLProject',
    version = '0.0.1',
    author = 'Sameer',
    author_email = 'sameer.cusat2019@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt'))