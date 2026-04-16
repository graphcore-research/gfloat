<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

## BUILDING

```
pip install -e .
( cd docs && make html )
# Install packages for testing - will install JAX, Torch, etc.
pip install -r requirements-dev.txt
pytest .

# Run tests on 32-bit linux/386 in Docker
bash etc/test-linux-386.sh
# Rebuild the cached linux/386 test image when dependencies change
GFLOAT_LINUX386_REBUILD=1 bash etc/test-linux-386.sh
```

#### Pushing
```
sh etc/package.sh
```
