tmpdir=/tmp/test-gfloat
mkdir -p $tmpdir

# Passing case
echo "Copyright (c) 2024 Graphcore Ltd. All rights reserved." > $tmpdir/t.sh
if sh etc/check-copyright.sh $tmpdir/t.sh
then
  echo PASS: Should have passed
else
  echo FAIL: Should have passed
fi

# Failing case
echo "Copyright (c) 2024 Graphcore Ltd. All rights xreserved." > $tmpdir/t.sh
if sh etc/check-copyright.sh $tmpdir/t.sh
then
  echo FAIL: Should have failed, but passed
else
  echo PASS: Should have failed
fi
