from setuptools import setup

setup(
   name='cs',
   version='0.0.1',
   description='A code for simulating neutron transport in slab or spherical geometry using the SN method for angle, and CS method for energy',
   author='Pablo Vaquer',
   packages=['cs'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
