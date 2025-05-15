# TEVA
The UVM Tandem Evolutionary Algorithm (TEVA), for intelligent feature importance and feature range importance 
identification.

## Installation
Installation of TEVA is very simple and requires no extra components other than the package itself (and its internal
dependencies).  To install, simply use `pip` in one of the following ways.

To access directly from git (note, this method sometimes has unforseen issues.  Method two is more reliable).  Replace
`<PAT>` with your GitHub Personal Access Token
```commandline
pip install git+https://<PAT>.github.com/TranscendEngineering/teva
```

To access through git clone.  Replace `<package_directory>` with the actual package location, and `<PAT>` with your
GitHub Personal Access Token
```commandline
cd <package_directory>
git clone https://<PAT>.github.com/TranscendEngineering/teva
```

```commandline
pip install <package_directory>
```

Finally, to install from a wheel file, download the wheel and locate its download location.  Then, run the following
command, replacing `<path_to_wheel>` with the actual filepath of the wheel:
```commandline
pip install <path_to_wheel>
```

To verify the installation, you should be able to import TEVA using the following python code:
```python
import teva
```

## Usage
The `teva` package is separated into two major subpackages, `ccea`, and `dnfea`.  `teva`, the Tandem
Evolutionary Algorithm, is an algorithm which uses two evolutionary algorithms in tandem, joining the functionality of
both `ccea` and `dnfea` to produce a more accurate and diverse result.

Each of the three algorithms may be used in on their own, but note that `dnfea` requires `ccea.ConjunctiveClause`
objects to run, so it is seldom used without first running `ccea`.

### TEVA
`teva` uses both the `ccea` and `dnfea` algorithms in series to produce more accurate and thorough results.  `teva`
requires an input matrix, or table of 'observations', where each observation is a row, and each column is a 'feature'.  
Each observation should also have a 'classification' value, that represents how that particular observation was
classified.  Any values are acceptable for classification or observation feature.  Unless otherwise specified, float
observation features will be interpreted to be Continuous, integer observation features will be interpreted to be
Ordinal, and all other types will be interpreted to be Categorical.

There are three major steps in the usage of the `teva` algorithm.  First, the algorithm must be initialized by 
instancing the `TEVA` class. Next, the algorithm must be prepared by 'fitting' the input data, using `teva.fit()`.  
Then, the algorithm will be run either in its entirety, or one classification at a time.

#### Step 1) Initialize:
```python
import numpy as np
import teva

n_observations = 1250
n_features = 5

# These observations and classifications should be replaced with your actual input data
observations = np.random.random(n_observations, n_features)
classifications = np.floor(observations.sum(axis=1))

# First, build the TEVA algorithm object:
# The following configuration inputs are required, for more optional inputs, see the documentation for teva.TEVA()
teva_alg = teva.TEVA(ccea_offspring_per_gen=10,
                     ccea_total_generations=30,
                     ccea_max_order=n_features,
                     ccea_layer_size=20,
                     ccea_n_age_layers=5,
                     ccea_fitness_threshold=np.log10(0.5),
                     
                     dnfea_total_generations=30,
                     dnfea_n_age_layers=5,
                     dnfea_gen_per_growth=3,
                     dnfea_max_order=10,
                     dnfea_layer_size=20)
```

#### Step 2) Fit Data:
```python
# Second, fit the observation data to the teva algorithm:
unique_classes = teva_alg.fit(observation_table=observations, 
                              classifications=classifications)
```

By default, the algorithm will interpret the feature types based on the input data.
To specify the feature types explicitly, you can do the following:
```python
feature_types = [
    teva.CONTINUOUS,
    teva.ORDINAL,
    teva.CATEGORICAL,
    teva.CONTINUOUS,
    teva.CONTINUOUS
]

# Second, fit the observation data to the teva algorithm:
unique_classes: list = teva_alg.fit(observation_table=observations, 
                                    classifications=classifications,
                                    feature_types=feature_types)
```

#### Step 3) Run Algorithm:
There are two ways to run the TEVA algorithm.  It can be run on all unique classes at once, or a single classifications
can be run at a time.  Examples of both options are shown below.

To run all unique classes at once:
```python
teva_alg.run_all_targets()
```

To run a single unique class:
```python
teva_alg.run(target_class=unique_classes[0])
```

To see an animated view of the algorithm as it runs in your console, enable the `visualize` flag in the run functions:
```python
teva_alg.run_all_targets(visualize=True)
```

#### Step 4) Get Results:
After a run, results will be stored in the TEVA archive, which can be accessed in a number of ways.

###### Return the data from the archive:
Get all the archived data:
```python
# returns a dictionary of {classification: list[ConjunctiveClause]}
ccs: dict[Any, list[teva.ConjunctiveClause]] = teva_alg.get_all_archived_ccs()

# returns a dictionary of {classification: list[DisjunctiveClause]}
dnfs: dict[Any, list[teva.DisjunctiveClause]] = teva_alg.get_all_archived_dnfs()
```

Get archived data for a specific target:

```python
ccs: list[teva.ConjunctiveClause] = teva_alg.get_archived_ccs(target_class=unique_classes[0])
dnfs: list[teva.DisjunctiveClause] = teva_alg.get_archived_dnfs(target_class=unique_classes[0])
```

###### Plot the data accuracy:
Plot all classes:
```python
teva_alg.plot_all()
```

Plot a single class:
```python
teva_alg.plot(target_class=unique_classes[0])
```

###### Export the data to Excel:
```python
ccea_filepath = "path/to/ccea/file.xlsx"
dnfea_filepath = "path/to/dnfea/file.xlsx"

teva_alg.export(ccea_filepath=ccea_filepath,
                dnfea_filepath=dnfea_filepath)
```

### CCEA
The CCEA algorithm takes observation data to generate `teva.ConjunctiveClause` objects.  Similarly to `teva`,
there are several steps to running the algorithm; initialization, data fitting, and data running.

#### Step 1) Initialize:

```python
import numpy as np
import teva

n_observations = 1250
n_features = 5

# These observations and classifications should be replaced with your actual input data
observations = np.random.random(n_observations, n_features)
classifications = np.floor(observations.sum(axis=1))

# First, build the TEVA algorithm object:
# The following configuration inputs are required, for more optional inputs, see the documentation for teva.TEVA()
ccea_alg = teva.CCEA(offspring_per_gen=10,
                     total_generations=30,
                     n_age_layers=5,
                     max_order=n_features,
                     layer_size=20,
                     fitness_threshold=np.log10(0.5))
```

#### Step 2) Fit Data:
```python
# Second, fit the observation data to the teva algorithm:
unique_classes = ccea_alg.fit(observation_table=observations, 
                              classifications=classifications)
```

By default, the algorithm will interpret the feature types based on the input data.
To specify the feature types explicitly, you can do the following:
```python
feature_types = [
    teva.CONTINUOUS,
    teva.ORDINAL,
    teva.CATEGORICAL,
    teva.CONTINUOUS,
    teva.CONTINUOUS
]

# Second, fit the observation data to the teva algorithm:
unique_classes: list = ccea_alg.fit(observation_table=observations, 
                                    classifications=classifications,
                                    feature_types=feature_types)
```

#### Step 3) Run Algorithm:
There are two ways to run the CCEA algorithm.  It can be run on all unique classes at once, or a single classifications
can be run at a time.  Examples of both options are shown below.

To run all unique classes at once:
```python
ccea_alg.run_all_targets()
```

To run a single unique class:
```python
ccea_alg.run(target_class=unique_classes[0])
```

To see an animated view of the algorithm as it runs in your console, enable the `visualize` flag in the run functions:
```python
ccea_alg.run_all_targets(visualize=True)
```

#### Step 4) Get Results:
After a run, results will be stored in the CCEA archive, which can be accessed in a number of ways.

###### Return the data from the archive:
Get all the archived data in one list:
```python
ccs: list[teva.DisjunctiveClause] = ccea_alg.get_all_archived_values()
```

Get archived data for a specific target, or separated by target:
```python
ccs: list[teva.DisjunctiveClause] = ccea_alg.get_archived(target_class=unique_classes[0])
```

### DNFEA
The DNFEA algorithm takes observation data and `teva.ConjunctiveClause` objects to generate `teva.DisjunctiveClause` 
objects.  The DNFEA algorithm also requires previously created or generated conjunctive clauses.  Typically, DNFEA will
be run after running CCEA.  Similarly to `teva`, there are several steps to running the algorithm; initialization, data 
fitting, and data running.

#### Step 1) Initialize:

```python
import numpy as np
import teva

n_observations = 1250
n_features = 5

# These observations and classifications should be replaced with your actual input data
observations = np.random.random(n_observations, n_features)
classifications = np.floor(observations.sum(axis=1))

# First, build the TEVA algorithm object:
# The following configuration inputs are required, for more optional inputs, see the documentation for teva.TEVA()
dnfea_alg = teva.DNFEA(total_generations=30,
                       gen_per_growth=3,
                       n_age_layers=5,
                       max_order=10,
                       layer_size=20)
``` 

#### Step 2) Fit Data:
```python
# Second, fit the observation data to the teva algorithm.  This requires a list of previously generated CCs:
unique_classes = ccea_alg.fit(observation_table=observations, 
                              classifications=classifications,
                              conjunctive_clauses=conjunctive_clauses)
```

#### Step 3) Run Algorithm:
There are two ways to run the DNFEA algorithm.  It can be run on all unique classes at once, or a single classifications
can be run at a time.  Examples of both options are shown below.

To run all unique classes at once:
```python
dnfea_alg.run_all_targets()
```

To run a single unique class:
```python
dnfea_alg.run(target_class=unique_classes[0])
```

To see an animated view of the algorithm as it runs in your console, enable the `visualize` flag in the run functions:
```python
dnfea_alg.run_all_targets(visualize=True)
```

#### Step 4) Get Results:
After a run, results will be stored in the DNFEA archive, which can be accessed in a number of ways.

###### Return the data from the archive:
Get all the archived data in one list:
```python
dnfs: list[teva.DisjunctiveClause] = dnfea_alg.get_all_archived_values()
```

Get archived data for a specific target, or separated by target:
```python
dnfs: list[teva.DisjunctiveClause] = dnfea_alg.get_archived(target_class=unique_classes[0])
```
