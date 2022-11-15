from setuptools import setup,find_packages
from typing import List

#Declaring variables for setup functions
PROJECT_NAME="MLops-BigSales"
VERSION="0.0.3"
AUTHOR="Manas Jaiswal"
DESCRIPTION="Practicing Mlops Project"
REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT="-e ."

def get_requirements_list() -> List[str]:
    """
    This function returns the list of requiremnets for the project in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list=[name.replace("\n","") for name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list

setup(
name=PROJECT_NAME,
version=VERSION,
author=AUTHOR,
description= DESCRIPTION,   
packages=find_packages(),
install_requires=get_requirements_list()
)            