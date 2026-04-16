<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

## BUILDING

```
pip install -e .
( cd docs && make html )
# Install packages for testing - will install JAX, Torch, etc.
pip install -r requirements-dev.txt
pytest .

# Run tests on 32-bit linux/386 in Docker
# Build image if missing, then run tests
bash etc/test-linux-386.sh
# Build image if missing
bash etc/test-linux-386.sh load
# Force rebuild image
bash etc/test-linux-386.sh build
# Run tests only (requires existing image)
bash etc/test-linux-386.sh run
```

#### Pushing
```
sh etc/package.sh
```
