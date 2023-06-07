# ML QUANTUM SEPARABILITY

The library is a toolbox written in python dedicated to the efficient generation of large labeled dataset for the quantum separability problem in high dimensional space. 
 

## dependencies
- numpy (>= 1.23.5)
- scipy (>= 1.10.0)

## organisation

- src : contain the algorithms
- data : contain the datasets used in our experiments
- models : contain all the models trained during our experiments


## usage

### Pipeline
the library is organised around the Pipeline class, which allow to define a sampling strategy as a serie of transformative steps applied to density matrices sampled from an initial probability distribution.

We give a typical use case in the following snipped of code :

```python
from types import save_dmstack, load_dmstack
from pipeline import *
from models.criteria import PPT
from models.approx_based import DistToSep
from transformers.sep_approximation import FrankWolfe

states, infos = Pipeline([
	('sample', InducedMeasure(k_params=[25]).states),		# induced measure of parameter 25
	('ppt only', select(PPT.is_respected, True)),			# respecting the PPT criterion
	('fw', add(FrankWolfe(1000).approximation, key = 'approx'), # compute the sep approx.
	('sel ent', select(DistToSep(0.01, sep_key = 'fw__approx').predict, Label.ENT))
]).sample(1000, [3,3])

save_dmstack('states_3x3', states, infos)
```
in this example, the following procedure is repeated until we obtain 1000 density matrices in dimensions [3,3] :

- we sample states from the induced distribution 
- we select the sampled states respecting the PPT criterion
- we add the separable approximation of each state in the infos dictionnary at the key 'fw__approx'
- we only select the sampled states at a distance 0.01 or greater of their approximation by Frank Wolfe.

the states and all the informations are then saved in the file 'states_3x3' at the .mat format and can be retrieved later via the function load_dmstack.

The pipeline function work with 3 types of functions :

 ### samplers
a sampler function produce a set of density matrices with relevant information and have the signature 
```python
def sampler(n_states : int, dims : list[int]) -> DMStack, dict
```
the following samplers can be found in the library :

- samplers.pure.RandomHaar
- samplers.mixed.RandomInduced
- samplers.mixed.RandomBures
- samplers.separable.RandomSeparable
- sampler.entangled.AugmentedPPTEnt

### transformer

a transformer function associate, to each density matrix in a set, ... 
a transformer function may use additional informations about the states and produce new informations.
```python
def transformer(states : DMStack, infos : dict) -> DMStack, dict
```
the following transformers can be found in the library :

- transformers.sep_approximations.FrankWolfe
- transformers.real_representation.GellMann
- transformer.real_representation.Measures

### model

a model function associate, to each density matrix in a set, a label.
a labeler function may use additional informations about the states and produce new informations.
```python
def labeler(states : DMStack, infos : dict) -> list[int], dict
```
the following labelxx can be found in the library :

- models.criteria.PPT
- models.criteria.SepBall
- models.criteria.Witnesses
- models.approx_based.MlModel
- models.approx_based.DistToSep
- models.approx_based.WitQuality

## datas

the datasets are grouped by :

- dimensions (3x3 or 7x7)
- usage (TRAIN or TEST)
- category (SEP, PPT, NPPT, FW)

The content of each files can be accessed by the function types.load_dmstack, which will return a DMStack containing all the state and a dictionnary containing informations about each states.
For states of the PPT category, the dictionary contain an approximation of the optimal witness in the 'fw__witness' key.

In all datasets, the states are in the form of complex density matrices.
Use transformations.GellMann or transformations.Measures to obtain a real-valued vector representation.

## models

the models are grouped by :

- dimensions of the input (3x3 or 7x7)
- creation method for the PPT-ENT examples (AUG or NOAUG)

the type of the model and the proportion of PPT-ENT states used during training is indicated in the file name :
for example the files

SVM_1000_[0.50]_(x)

(with x an index in [0,4]) contain a SVM trained using a dataset of 1000 examples per class where 50% of the entangled examples were PPT-ENT.

All the models are accessible by the function joblib.load in the form of a GridSearchCV model (from sklearn).
All the models in the library use the Gell-Mann representation of states as input.
