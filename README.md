# Recurrent Poisson Factorization (RPF)
![Build Status](https://img.shields.io/teamcity/codebetter/bt428.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

RPF is a temporal recommender system which is able to recommend the right item to the right user at the right time by utilizing the recurring temporal patterns in user-item engagement. The repository contains the implementation of different variants of RPF, i.e. Hierarchical RPF, Social RPF, Dynamic RPF, Dynamic Social RPF.


## Prerequisites

- Matlab version R2014a or later

## Features

-  A coherent generative model for user-item interaction over time

- The Social RPF model is able to infer the user interests on different items using her social relations.

- Dynamic RPF considers the variable interest of user over time

- Hierarchical RPF is able to consider the diversity of users interests and items popularity using a hierarchical structure.

- A fast variational algorithm for inference on the proposed time-dependent model.

## Data

The input format for the events is as follows:
```
unixTime userId    itemId
```
The events should be sorted in an increasing order of time. The userId and itemIds are sequential Integer numbers starting from 1. The name of this file should be datasetName.tsv .

Social Methods such as SRPF and DSRPF takes an extra input file which contains the adjacency list among the users. The name of this file should be datasetName\_adjList.txt. Each line of this file starts with id of a user and then the number of users that she follows and then the list of users that she follows:

```
userId1   N user_1 user_2 ... user_N
```
The LastFM dataset which is used in the RPF paper is in the Dataset folder as a sample.
## Running The Code

In order to run each of the HRPF, DRPF, SRPF and DSRPF

- Go to the methods folder

- Set the Dataset in the "run" Script

- Run the run script


The results will be saved under the "Results" folder.

## Citation 

In case of using the code, please cite the following paper:

Hosseini, Seyed Abbas, et al. "Recurrent Poisson Factorization for Temporal Recommendation." arXiv preprint arXiv:1703.01442 (2017).
