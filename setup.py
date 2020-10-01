import setuptools


setuptools.setup(
    name="tensor_dyve",
    version="1.3",
    author="Catalin Herghelegiu",
    author_email="cherghelegiu11@gmail.com",
    description="Train, evaluate, export user defined machine learning models.",
    entry_points={
        "console_scripts": ['tensor_dyve = TensorDyve.tensor_dyve:main']
        },
    packages=setuptools.find_packages(),
    install_requires=[ # have not tested with tf 2.x and it's import namespace changes for backwards compat

          'tensorflow-gpu>=1.12.3,<=1.15.3', 'opencv-python', 'imgaug', 'psutil', 'tqdm'
      ],

)
