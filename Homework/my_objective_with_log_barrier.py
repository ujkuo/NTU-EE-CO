#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Version 
#   Author: ujkuo
#   GitHub: github.com/ujkuo
#   Copyleft (C) 2021 ujkuo All rights reversed.
#

import numpy
import math

def quadratic(P, y, r, x):
	return numpy.dot(numpy.dot((x-y).T, P), (x-y))/2 + r

def my_objective_with_log_barrier(P, y, r, x, t):
    # gradient
    tx = 0.0
    x3 = 0.0
    for i in range(3):
        tx += numpy.dot(P[i], (x[:2] - y[i])) / (-quadratic(P[i], y[i], r[i], x[:2]) + x[2])
        x3 += 1 / (-quadratic(P[i], y[i], r[i], x[:2]) + x[2])

    g = numpy.array([tx[0], tx[1], t-x3])
    
    # hessian
    s = (3,3)
    h = numpy.zeros(s, dtype = float)
    for i in range(3):
        difference = numpy.dot(P[i], (x[:2] - y[i]))
        delta = -(quadratic(P[i], y[i], r[i], x[:2])) + x[2]
        delta_square = delta * delta
        h[0:2, 0:2] += (P[i] * delta + numpy.dot(difference[:, numpy.newaxis], difference[:, numpy.newaxis].T)) / delta_square
        h[2, 2] += 1 / delta_square
        h[:2, 2] -= difference / delta_square
        h[2, :2] -= difference / delta_square

    phi0 = 0.0
    phi1 = 0.0
    phi2 = 0.0
    phi0 -= numpy.log(-quadratic(P[0], y[0], r[0], x[:2]) + x[2])
    phi1 -= numpy.log(-quadratic(P[1], y[1], r[1], x[:2]) + x[2])
    phi2 -= numpy.log(-quadratic(P[2], y[2], r[2], x[:2]) + x[2])
    phi = phi0 + phi1 + phi2

    print("\n=======================================")
    print('\nShow the result of the function my_objective_with_log_barrier: \n')
    print("The value of function f at point x is ", t*x[2] + phi, '\n')
    print("The Gradient of function f at point x is \n", g, '\n')
    print("The Hessian of function f at point x is \n", h, '\n')
    v = t*x[2] + phi

    return v, g, h

def domain_exist(x):
    for i in range(3):
        if (-quadratic(P[i], y[i], r[i], x[:2]) + x[2]) <= 0:
            return True
    return False

def print_results(iteration, inner_steps_record, t_record, value_record, newton_index, x_record):
    print("\n===========================")
    print("\nThe result of Newton Method")
    print("\n===========================")
    print("\nThe number of outer interation is : ", iteration)
    print("\nFor each outer iteration, the number of inner Newton steps is : ", inner_steps_record)
    print("\nFor each outer iteration, t and the function value at the end of the outer iteration is :")
    print("\nt = ", t_record)
    print("\nf = ", value_record)
    print("\nThe total number of Newton steps is : ", newton_steps_record)
    print("\nThe optimal point x*(t) at the end of the outer iteration is : ", x_record)


numpy.seterr(invalid='print')

# Parameters
P = numpy.array((([2, 1], [1, 2]), ([2, 0], [0, 3]), ([2, -1], [-1, 3])))
y = numpy.array((([1, 0]), ([-1, 2]), ([-1, -2])))
r = numpy.array([0, 1, -1])
x0 = numpy.array([0, 0, 10])
t = 1

# Show the function my_objective_with_log_barrier
v = 0.0
g = 0.0
h = 0.0
v, g, h = my_objective_with_log_barrier(P, y, r, x0, t)
print("\nmy_objective_with_log_barrier : ", v, g, h)

# Implement Newton Method
# Parameter setting
alpha = 0.1
beta = 0.7
mu = 2000
newton_index = 0
iteration = 0
newton_step = numpy.dot(-numpy.linalg.inv(h), g)
newton_decrement = (-numpy.dot(g.T, newton_step))**0.5
print("\nstep and decrement:" ,newton_step, newton_decrement)

# Set criterion
criterion = True

# Set epsilon
inner_epsilon = 1e-5
outer_epsilon = 1e-8

# Set recorder
inner_steps_record = []
value_record = []
t_record = []
newton_steps_record = []
x_record = []

while criterion == True:
    print("start")
    inner_steps = 0
    # vt = newton_decrement * newton_decrement / 2
    while newton_decrement * newton_decrement / 2 > inner_epsilon:
        print("comparison1 enter")
        s = 1
        x_next = x0 + newton_step * s

        # checking domain
        while domain_exist(x_next):
            s *= beta
            x_next = x0 + newton_step * s
        print("domain check pass")
        flag = True
        #while flag == True:
        #    for i in range(3):
        #        check = -(quadratic(P[i], y[i], r[i], x_next[:2])) + x_next[2]
        for i in range(3):
            check = -(quadratic(P[i], y[i], r[i], x_next[:2])) + x_next[2]
            if check <= 0:
                flag = True
            else:
                flag = False
        #if flag == True:
        #    s *= beta
        #    x_next = x0 + newton_step * s

        v_next, g1, h1 = my_objective_with_log_barrier(P, y, r, x_next, t)
        value = v - alpha * s * newton_decrement * newton_decrement
        while v_next > value:
            s *= beta 
            x_next = x0 + newton_step * s
            v_next, g2, h2 = my_objective_with_log_barrier(P, y, r, x_next, t)
        print("comparison2 pass")
        
        # update
        x0 = x0 + s * newton_step
        v, g, h = my_objective_with_log_barrier(P, y, r, x0, t)
        newton_step = numpy.dot(-(numpy.linalg.inv(h)), g)
        newton_decrement = (-numpy.dot(g.T, newton_step))**0.5
        newton_index += 1
        inner_steps += 1
        print("update pass")

    print("inner_epsilon pass")
    
    # check criterion
    if (3/t) > outer_epsilon:
        criterion = True
    else:
        criterion = False
    print("criterion check pass")

    # record
    inner_steps_record.append(inner_steps)
    value_record.append(v)
    t_record.append(t)
    newton_steps_record.append(newton_index)
    x_record.append(x0)

    # update parameters
    t *= mu
    iteration += 1
    v, g, h = my_objective_with_log_barrier(P, y, r, x0, t)
    newton_step = numpy.dot(-(numpy.linalg.inv(h)), g)
    newton_decrement = (-numpy.dot(g.T, newton_step))**0.5

    print("end")

# Print results
print_results(iteration, inner_steps_record, t_record, value_record, newton_steps_record, x_record)






















