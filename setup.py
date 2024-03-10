from setuptools import find_packages, setup
from typing import List

Hyp_e_dot = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''this function returns the requirements from txt file '''
    requirements = []
    with open(file_path)as file_obj:
        requirements = file_obj.readlines()
        requirements = [x.replace('\n','') for x in requirements]
        if Hyp_e_dot in requirements:
            requirements.remove(Hyp_e_dot)
    return requirements        


setup(
    name='HousePricePrediction',
    version='0.0.1',
    author='BelhsanHmida',
    author_email='mohamedbelhsanhmida@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)