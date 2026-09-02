from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='p3dformer',
        version='0.1.0',
        description='Review-stage code for 3D instance segmentation',
        packages=find_packages(include=['p3dformer', 'p3dformer.*', 'lib', 'lib.*']),
        python_requires='>=3.9',
    )
