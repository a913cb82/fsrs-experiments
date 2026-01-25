#!/bin/bash
set -e

# Specific commit hashes for stability
CORE_HASH="ec5d1c729de35e0cc7f9482a8acd5ae70d1b574e"
BINDING_HASH="47f0a78d8c8ae1c3df8c04700e555698948f8aa2"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting FSRS Rust Backend Setup...${NC}"

# 1. Clone and Checkout Core
if [ ! -d "fsrs-rs-repo" ]; then
    echo "Cloning fsrs-rs core..."
    git clone https://github.com/open-spaced-repetition/fsrs-rs fsrs-rs-repo
    cd fsrs-rs-repo && git checkout $CORE_HASH && cd ..
fi

# 2. Clone and Checkout Bindings
if [ ! -d "fsrs-rs-python-repo" ]; then
    echo "Cloning fsrs-rs-python bindings..."
    git clone https://github.com/open-spaced-repetition/fsrs-rs-python fsrs-rs-python-repo
    cd fsrs-rs-python-repo && git checkout $BINDING_HASH && cd ..
fi

# 3. Apply patches
echo "Applying FSRS-6 compatibility patches..."
cd fsrs-rs-python-repo
# Use --ignore-whitespace to be more robust
git apply ../patches/fsrs_rs_python_v6.patch
cd ..

echo -e "${GREEN}Repos cloned, checked out to stable commits, and patched successfully.${NC}"
echo "To finish installation, run:"
echo "  cd fsrs-rs-python-repo"
echo "  pip install maturin"
echo "  maturin develop --release"