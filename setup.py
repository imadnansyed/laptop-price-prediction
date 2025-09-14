## this helps us to install our local packages like src etc
## so that we can run the code from anywhere
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)-> List[str]:
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

setup(
    name="laptop_price_predictor",
    version="0.0.1",
    author="Adnan",
    author_email="asaid31620@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
)