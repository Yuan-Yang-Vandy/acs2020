# Computational Models for Solving Raven's Progressive Matrices Using Analogies and Image Transformations

This repository contains the source code for the ACS 2020 paper **Computational Models for Solving Raven's Progressive Matrices Using Analogies and Image Transformations**.

## 1. Requirements
* `Python 3.7` is used for this project.
* Package dependencies reside in `requirements.txt`.

## 2. Project Structure
* **Main entry** is `main.py`.
* **Inputs** are in folder `problems`, including all the standard RPM problems, and the coordinates files to segment them into matrix entries and options.
* **Outputs**, including reports of confident, neutral and prudent strategies, are in folder `reports`. Every time you run the code, it will generate new time-stamped reports in this folder.
* **Detailed Results** are in the folder `data`. For each strategy and each problem, there are 3 files in this folder `<strategy>_<problem>.json`, `<strategy>_<problem>_prediction.png` and `<strategy>_<problem>_selection.png`, which are detailed computational info, predicted image and the selected option.
* **Similarity Values** are pre-computated and stored in the folder `precomputated-similarities` because it takes one or two days to compute all the similarity values.

## 3. Running
It takes roughly **1 hour** on my desktop for a run of the code.


