# safe-squeezing

This repository contains numerical procedure to compute antisparse representation with / without safe squeezing [1].
We elaborate on the notion of safe squeezing in

> [1] Clément Elvira, Cédric Herzet: “Safe squeezing for antisparse coding”, arXiv, october 2019; [arXiv:1911.07508](http://arxiv.org/abs/0000.00000)

The above paper contains theoretical results and several applications that can be reproduced with this toolbox.

This python toolbox is currently under development and is hosted on Gitlab. If you encounter a bug or something unexpected please let me know by [raising an issue](https://gitlab.inria.fr/celvira/safe-squeezing/issues) on the project page.

# Requirements

safe-squeezing works with python 3.5+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)
 -   [Matplotlib](http://matplotlib.org)


# Install from sources

1. Clone the repository
```bash
git clone https://gitlab.inria.fr/celvira/safe-squeezing.git
```

2. Enter the folder
```bash
cd safe-squeezing
```

3. (Optional) Create a virtual environment and activate it
```bash
virtualenv venv -p python3
source venv/bin/activate
```

4. Install the dependencies
```bash
pip install -r requirements.txt
```

5. And execute `setup.py`
```bash
pip install .
```
or 
```bash
pip install -e .
```
if you want it editable.

## Running the experiments

```bash
cd experiments/TSP2019
python exp.py
python vizu.py
```
or
```bash
python python vizu.py --save
```
if you want to save the plot.


## Running the demo examples

```bash
cd notebook
jupyter notebook
```


# Licence

This software is distributed under the [CeCILL Free Software Licence Agreement](http://www.cecill.info/licences/Licence_CeCILL_V2-en.html)


# Cite this work

If you use this package for your own work, please consider citing it with this piece of BibTeX:

```bibtex
@article{Elvira2019arxiv,
	Author = {Elvira, Cl\'ement and Herzet, C\'edric},
	Journal = {Available at \url{http://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Publications_files/Elvira2020a.pdf}},
	Month = {october},
	Pdf = {http://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet/Publications_files/Elvira2020a.pdf},
	Title = {Safe squeezing for antisparse coding},
	Year = {2019},
}
```