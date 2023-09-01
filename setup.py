from setuptools import setup, find_packages
setup(
    name="Lung Disease Classification",
    version="0.0.1",
    description="Classifies and segments disease that affects the lungs by using Xrays for analysis",
    install_requires=['tensorflow','pandas','numpy'],
    author="Randy Kofi Ansah",
    author_email="randyansah97@gmail.com",
    packages=find_packages()

)