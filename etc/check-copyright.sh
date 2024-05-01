#!/usr/bin/bash
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

PATTERN='Copyright \(c\) 202[0-9] Graphcore Ltd\. +All rights reserved\.'
IGNORE=(docs/Makefile docs/make.bat \*.md \*requirements\*.txt)
GITNOTMATCH=':(exclude)' # See git pathspec docs
git ls-files -- . ${IGNORE[@]/#/$GITNOTMATCH}  | xargs grep -L -E "$PATTERN"
