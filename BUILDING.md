<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

## BUILDING

```sh
pip install -e .
( cd docs && make html )
# Install packages for testing - will install JAX, Torch, etc.
pip install -r requirements-dev.txt
pytest .
```

To run tests on 32-bit linux/386 in Docker:
```sh
# Build image if missing, then run tests
bash etc/test-linux-386.sh
# Build image if missing
bash etc/test-linux-386.sh load
# Force rebuild image
bash etc/test-linux-386.sh build
# Run tests only (requires existing image)
bash etc/test-linux-386.sh run
```

#### Packaging for pypi release:

Edit version number in
```sh
sh etc/package.sh
```
