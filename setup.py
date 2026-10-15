from setuptools import setup, find_packages

setup(
    name="loan_default_model_package",
    version="0.0.1",
    author="BH D M V S S SARMA",
    description="Loan Default ML pipeline packaged with Random Forest",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ]
)