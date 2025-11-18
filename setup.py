from setuptools import setup, find_packages

setup(
    name="project_pipeline",
    version="0.1.0",
    description="Machine Learning Pipeline for Network essentials Project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "imbalanced-learn>=0.9.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "tensorflow>=2.10.0",
        "keras-tuner>=1.1.3",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "joblib>=1.2.0"
    ],
    python_requires=">=3.8",
)
