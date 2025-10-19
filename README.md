# H-AIF

## Installation
UV installation
```
# curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# wget  
wget -qO- https://astral.sh/uv/install.sh | sh
```

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules https://github.com/ldddddddl/H-AIF.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```


```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
# Options
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.