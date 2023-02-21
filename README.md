# hb-mep

## Setup Instructions

### Move into top-level directory
```
cd hb-mep

```

### Create environment
```
python3 -m venv .venv

```

### Activate environment
```
source .venv/bin/activate

```

### Install dependencies
```
pip install -r requirements.txt

```

### Install package
```
pip install -e src/hb-mep

```

### Install jupyter kernel
```
python -m ipykernel install --user --name=hb-mep-ipython

```

### Run jupyter server
```
jupyter notebook notebooks/

```

You can now use the kernel `hb-mep-ipython` to run notebooks.