#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Version 
#   Author: ujkuo
#   GitHub: github.com/ujkuo
#   Copyleft (C) 2021 ujkuo All rights reversed.
#

import numpy
import cvxpy

# Parameters
P = numpy.array((([2,1], [1,2]), ([2,0], [0,3]), ([2,-1], [-1,3])))
y = numpy.array((([1,0]), ([-1,2]), ([-1,-2])))
r = numpy.array([0,1,-1])

# Variables
x = cvxpy.Variable(2)
w = cvxpy.Variable(1)

# Solving problem
problem = cvxpy.Problem(cvxpy.Minimize(w),
            [cvxpy.quad_form(x-y[0], P[0])/2 + r[0] - w <= 0,
             cvxpy.quad_form(x-y[1], P[1])/2 + r[1] - w <= 0,
             cvxpy.quad_form(x-y[2], P[2])/2 + r[2] - w <= 0])
problem.solve()

# Print
print("The optimal p* is ", problem.value, ", and the solution x* is ", x.value)

