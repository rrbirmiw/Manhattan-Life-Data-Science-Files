
"""Loss_functions contains function definitions for the loss and gradient
   functions of:
   -huberized(smooth) hinge loss
   - squared hinge loss

   Used in conjunction with linearsvm.py

Written by Rahul Birmiwal
2018
"""


import numpy as np

def squared_hingeloss_obj(z):
    """ Objective function for squared hinge loss"""
    if (z >= 1):
        return 0
    else:
        return (1-z)**2

def squared_hingeloss_grad(z):
    """ Gradient function for squared hinge loss"""
    if (z >= 1):
        return 0
    else:
        return -2*(1-z)

def huber_loss_obj(z):
    """ Objective function for Huberized hinge loss"""
    if (z >= 1):
        return 0
    if ( z > 0 and z < 1):
        return 0.5*((1-z)**2)
    if (z <= 0):
        return 0.5 - z

def huber_loss_grad(z):
    """ Gradient function for huberized hinge loss"""
    if (z > 1):
        return 0
    if ( z >= 0 and z <= 1):
        return z-1
    if (z < 0):
        return -1
