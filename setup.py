from setuptools import setup, find_packages

setup(
    name="ns-aime-xai2026",
    version="0.2.0",
    description="NS-AIME: Rule-Guided Re-Constraint of Approximate Inverse Explanations",
    author="Takafumi Nakanishi",  
    packages=find_packages(),
    # Core deps are intentionally minimal (numpy + matplotlib only).
    # scipy is optional (sparse inputs); scikit-learn / lightgbm are only
    # needed to run the examples, not the library itself.
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.0.0",
    ],
    extras_require={
        "sparse": ["scipy>=1.7.0"],
        "examples": ["scikit-learn>=1.0.0", "pandas>=1.3.0", "lightgbm>=3.0.0"],
    },
    python_requires=">=3.8",
    license="Research Only (Commercial License required for business use)",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)