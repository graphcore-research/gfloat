# Set version numbers, make package, and publish

set -o errexit

VERSION="0.0.5"
perl -pi -e 's/^(release|version) = "([\d.]+)"/$1 = "'$VERSION'"/' docs/source/conf.py
perl -pi -e 's/^version = "([\d.]+)"/$1 = "'$VERSION'"/' pyproject.toml

( cd docs && make html )

rm -rf dist
pip install build twine
python -m build
echo "Enter PyPI API Token"
echo __token__ | twine upload --repository pypi dist/* --verbose
