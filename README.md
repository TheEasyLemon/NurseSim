# Nurse Simulation

Dawson Ren, November 14th, 2022

See the attached PDF for some initial thoughts regarding the problem. See below for documentation for testing.

## Testing Files

We provide test cases as a text file under the tests subfolder for every directory in `src`. The format of the text file is given below.
```
<P matrix>

<Q matrix>

<r vector>

<Y matrix>
```
Each matrix is delimited with commas. Each row is delimited by newlines. There is a break between every matrix/vector. The number of nurses and open shifts is inferred from the `P` matrix, and violations will be raised in the middle of parsing the file.

To run a unit test, we use the standard library `unittest` module. From the top-level directory, simply use `python3 src/ProblemReader/tests/ProblemReader_Tests.py`.
