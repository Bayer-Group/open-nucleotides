from setuptools import setup

setup(
    name="nucleotides",
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "nucleotides/_version.py",
    },
)
