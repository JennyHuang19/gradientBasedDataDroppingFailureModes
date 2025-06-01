# gradientBasedDataDroppingFailureModes

This repository contains code and experiments for auditing the robustness of model conclusions to worst-case data removal. A central concern in data analysis is whether dropping a very small subset of data can significantly change the outcome of a model, posing risks to generalization and trustworthiness. Identifying such data points directly is computationally intractable, even for simple models like OLS regression.

Recent works have proposed a variety of approximation methods to detect this form of non-robustness. In this project, we systematically evaluate these approaches and uncover critical cases where popular gradient-based data dropping approximations fail to detect non-robustness in both synthetic and real-world datasets. Our study reveals that:

- Several widely used approximations fail to identify non-robustness.

- A simple recursive greedy algorithm consistently succeeds across all test cases.

- This greedy approach is also computationally efficient—often significantly faster than its competitors.

Code in this repo reproduces all key experiments in the paper
**[Approximations to Worst-Case Data Dropping: Unmasking Failure Modes](https://arxiv.org/abs/2408.09008)**.

