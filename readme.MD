# Optimization and deployment challenges of closed-source LLMs for clinical note abbreviation expansion

## Overview

This is the codebase and paper prepared by the author for a university project.

The project requirement was to apply natural language processing to a health application.

My paper explores the ability of OpenAI models to perform clinical acronym expansion.

My findings and discussion can be found in the report.pdf file.

## Set-up

All functions are run through the notebook 'main.ipynb'.

Substantive code is imported from .py files in the /utils folder.

Data is included or placeholdered in the /data folder. Note:
- 'AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt' from the original Moon 2014 paper (available: https://conservancy.umn.edu/handle/11299/137703)
- 'labeled_sf_lf_map_DC.csv' from Adams, 2020, with minor amendment (original available: https://github.com/griff4692/LMC/tree/master/shared_data/casi)
- 'preprocessed_dataset_window_10.csv' from Adams, 2020 (available: https://github.com/griff4692/LMC/tree/master/shared_data/casi)

To run the API calls, an OpenAI API key needs to be inputted.

Python requirements include: nltk, pandas, datetime, tqdm, openai, sklearn.
