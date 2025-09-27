Find the report [here]([https://forms.gle/giucmSFyYQiBjL1S6](https://drive.google.com/file/d/1MC-PP57bvYaZQTbKqghoQttn2cJBnSaL/view?usp=sharing)).
```
# ML Competitions: Models & Scoring

## Competition Submission Form

Once everything below is completed, every team should submit the
[Competition Submission Form](https://forms.gle/giucmSFyYQiBjL1S6). It is
due Wednesday, April 9th, at 11:59PM.

## Repository Instructions

This is the base repository to build upon.

1. Fork this repository, once per group
2. Ensure your repository is set to private
3. Add @ryanhammonds and @ustunb to the repository
4. Add all team members to the respository

Then install the repostory code in your environment:

```bash
git clone https://github.com/{your fork}/mlc.git
pip install -e mlc
```

## Submission Template Instructions

There is `ScorableModelTemplate` class and `compute_score` function
for each comptition, accessible from:

- `mlc/birdclef.py`
- `mlc/bugnist.py`
- `mlc/cashflow.py`

Extend the `ScorableModelTemplate` and include your work. The `.process_inputs` and `.predict`
methods need to be implemented, and have specific required inputs. See the docstring of
these methods, specific to your compeititon, to ensure correct inputs are passed.
In BugNIST and BirdCLEF, these functions require a list of file paths to each tif or ogg file.
Cashflow reqiures two inputs, string paths to consumer_data.parquet and transactions.parquet.

The `.process_inputs` method should:

1) Read in raw data
2) Compute features (optional for the first assignment)

The `.predict` method should:

1) Call `.process_inputs` to read in data and compute features
2) Pass computed features to trained, e.g. pytorch or sklearn, model
3) Return predictions that are compatible with you competitions `compute_score` function.

Each competition has a specific `compute_score` function. Check it to ensure what is
returned from `.predict` is compatible.

### __check_rep__

Each `ScorableModelTemplate` class has a `__check_rep__` method that tests if your predictions
can be scored. This method runs tests when your template sub-class is initialized.
See the `__check_rep__` method for:

- an example of how to pass the output from `.predict` to `compute_score`, specific to each competition
- the tests that are required to pass

### Submission

Complete the following in either a submission.py or submission.ipynb. Commit this file and push it to
your github fork. It is what will be used for grading.

```python
from mlc.[your competition] import ScorableModelTemplate

class ScorableModel(ScorableModelTemplate):

    def predict(self, raw_files: list[str]):
        """Input argument will vary. See you competition's template.

        :param raw_files: list of file path strings, depends on competition
        :return predictions: dataframe or np.array, depends on competition
        """
        # Implement this: may return random predictions for the first assignment
        raise NotImplementedError()

    def process_inputs(self, raw_files: list[str]):
        """Input argument will vary. See you competition's template.

        :param raw_files: list of file path strings, depends on competition
        :return: anything needed for you model to make predictions, e.g. features or processed data
        """
        # Implement this: only need to read in files for first assignment
        raise NotImplementedError()

# Intialize, runs: __check_rep__ to validate class
model = ScorableModel() # error will be raised if the above is not implemented correctly


## DSMLP

[Slides](https://docs.google.com/presentation/d/1NAEO91toHvFN9y_7jojfs3pxln0MkH4YhGTNdSVj6xU/edit?usp=sharing)
```
on DSMLP.

- How to login
- How to access datasets on DSMLP
- How to clone you forked version of this repository to DSMLP
- How to start notebooks with and without GPU
