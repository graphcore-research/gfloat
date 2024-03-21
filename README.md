# gfloat: Generic floating-point types in Python

An implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754): Binary16, Binary32
 - [OCP Float8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf): E5M2, E4M3
 - [IEEE WG P3109](https://github.com/awf/P3109-Public/blob/main/Shared%20Reports/P3109%20WG%20Interim%20report.pdf): P{p} for p in 1..7

See https://gfloat.readthedocs.io for documentation.

## BUILDING

```
pip install -e .
cd docs 
make html
cd ..
```

#### Pushing
```
rm -rf dist
python3 -m build
echo __token__ | twine upload --repository pypi dist/* --verbose
```

#### Notes

All NaNs are the same, with no distinction between signalling or quiet, 
or between differently encoded NaNs.

