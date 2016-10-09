#!/usr/bin/env python
import sys
import pytest

if __name__ == '__main__':
    args = ['-vrxs', '--cov', 'photomosaic', 'test.py']
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    sys.exit(pytest.main(args))
