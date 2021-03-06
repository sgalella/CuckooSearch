# Cuckoo Search
Implementation of the [cuckoo search algorithm](https://en.wikipedia.org/wiki/Cuckoo_search) proposed in [**Cuckoo Search via Lévy Flights**](https://ieeexplore.ieee.org/document/5393690):

>"It was inspired by the obligate brood parasitism of some cuckoo species by laying their eggs in the nests of other host birds (other species). [...] If a host bird discovers the eggs are not their own, it will either throw these aliens eggs away or simply abandon the nest and build a nest elsewhere." — From [Wikipedia](https://en.wikipedia.org/wiki/Cuckoo_search)

<p align="center">
    <img width="512" height="304" src="images/cuckoo_search.gif">
</p>

## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

If using Conda, you can also create an environment with the requirements:

```bash
conda env create -f environment.yml
```

By default the environment name is `cuckoo-search`. To activate it run:

```bash
conda activate cuckoo-search
````



## Usage

Run the algorithm from the command line with:

```bash
python -m cuckoo_search
```

To see the different visualization modes check `notebooks/`.

