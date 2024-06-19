<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

## BUILDING

```
pip install -e .
( cd docs && make html )
```

If notebook outputs have changed, then verify the new outputs are correct, and run
```
pytest --nb-force-regen
```
If a notebook has volatile outputs (e.g. timings), see the `nbreg/diff_replace` metadata in `04-benchmark.ipynb` for an example of how to selectively ignore it.


#### Pushing
```
sh etc/package.sh
```
