from setuptools import setup, find_packages

with open('README.md', 'r') as fp:
    long_description = fp.read()

setup(
    name='ikfs-anomaly-detector',
    version='1.0.1',
    description='Программная система обнаружения аномалий в телеметрии бортового фурье-спектрометра ИКФС-2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: MacOS X',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Russian',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='ikfs anomaly lstm',
    url='https://github.com/DSPLab-IC6/ikfs_anomaly_detector',
    author='Anton Telyshev',
    author_email='anton.telishev@yandex.ru',
    license='MIT',
    packages=find_packages(),
    package_data={
        'ikfs_anomaly_detector': [
            'models/**/*.h5',
        ],
    },
    entry_points={
        'console_scripts': ['ikfs-anomaly-detector=ikfs_anomaly_detector.__main__:main'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'dataclasses',
        'h5py',
        'keras',
        'matplotlib',
        'mock',
        'numpy',
        'pydot',
        'tensorflow',
    ],
    include_package_data=True,
    zip_safe=False,
)
