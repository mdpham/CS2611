# For project evaluation:
The file `aclu_opinions.csv` is a subsample of the [Kaggle SCOTUS dataset](https://www.kaggle.com/datasets/gqfiddler/scotus-opinions) from the list of cases in `aclu_cases.csv` and so you don't need to download the latter in order to pre-process it for embedding using `aclu_dataset_preprocessing.ipynb` since this is already done.

The directory `report` contains images generated from the Jupyter Lab notebooks for the written final report.

The directory `src` includes a `ProjectUtils` class containing all relevant methods for the project (preprocessing, dimension reduction, plotting).

The main file of interest is `aclu_dataset_embedding.ipynb` where the dataset is loaded, embedded and tested.
The resulting values were reported in the submitted file.

The notebook `exploration.ipynb` was used for preliminary work and may be ignored (most of its content is reproduced in `aclu_dataset_embedding.ipynb`).
