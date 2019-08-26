## Solve Convex Problem  

`diffcp` is a Python package [https://github.com/cvxgrp/diffcp] 

We make use of the fast projection in the homogenous cone for optimal solution. Then we inject noise and leverage the SGD or Proximal mathed. 


##### Dependency requirements:
* [cvxpy](http://www.cvxpy.org/en/latest/)
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [SCS](https://github.com/bodono/scs-python)
* Python >= 3.5+

For Splitting Cone and Homogeneous Self Dual Embedding, check the papers by
 * [B. O'Donoghue, E. Chu, N. Parikh, and S. Boyd](https://web.stanford.edu/~boyd/papers/scs.html) 
 * [A. Agrawal, S. Barratt, S. Boyd, E. Busseti, and W. Moursi](https://web.stanford.edu/~boyd/papers/diff_cone_prog.html)
 
