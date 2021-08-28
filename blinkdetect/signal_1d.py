#!/usr/bin/env python3
# coding: utf-8


from typing import List, Union
from matplotlib.pyplot import step
import numpy as np
import random
from math import floor, ceil

def prolong(y: Union[np.ndarray, List],x, upsampling=True, rand=True, min_len=6, max_len=21, Num=None):
    """
    interpolate y and x
    accept numpy integer
    """

    blink_len = len(y)
    if upsampling:
        _NUM = random.randint(blink_len,max_len)#len(y)*2 # to be computed automatically
    else:
        _NUM = random.randint(min_len,blink_len)

    # checking numpy type
    if isinstance(Num,int) or isinstance(Num,np.integer):
        _NUM = Num
        
    xvals = np.linspace(x[0], x[-1], _NUM)
    yinterp = np.interp(xvals, x, y)
    
    return yinterp, _NUM

def extend(y: Union[np.ndarray, List], x: Union[np.ndarray, List], num:int=30, val=None):
    """get a 1d signal and a linspace, extend the linspace length to be total num to the left
    and the right

    Args:
        y (Union[np.ndarray, List]): 1d signal
        x (Union[np.ndarray, List]): linespace where the signal ought to be extended
        num (int, optional): number of samples. Defaults to 60.
    """
    _extension_len = num - len(x)
    _extension_half = _extension_len/2

    _extension_left = floor(_extension_half)
    _extension_right = ceil(_extension_half)

    far_left = int(x[0])-_extension_left # -2
    far_right = int(x[-1])+_extension_right+1 # 7
    # far_left = x[0]-_extension_left-1 # -2
    # far_right = x[-1]+_extension_right # 7
    padd_right = 0
    padd_left = 0 # 2
    _middle_left = int(x[0])
    _middle_right = int(x[-1])

    if far_right >= len(y):
        padd_right = far_right-len(y)
        far_right = len(y)
    if far_left < 0:
        padd_left = abs(far_left)
        far_left = 0


    res = np.concatenate((y[far_left:_middle_left], y[_middle_left:_middle_right], y[_middle_right:far_right]))
    if type(val) in [float, int]:
        res = np.pad(res, (padd_left,padd_right), constant_values=val)
    else:
        res = np.pad(res, (padd_left,padd_right), mode='edge')

    return res, np.linspace(far_left-padd_left, far_right+padd_right-1, len(res))

def shift(x: Union[np.ndarray, List],steps:Union[int, List]):
    """shift a singal represented as numpy.ndarray, edge points padded.

    Args:
        x (Union[np.ndarray, List]): [description]
        steps (int): shift (int, optional): positive means shifting to the right, padding to left, and vice versa

    Returns:
        numpy.ndarray: shifted and padded signal
    """
    padd_left = 0
    padd_right = 0

    if type(steps) is list and len(steps) == 2:
        _steps = random.randint(steps[0], steps[1])
    elif type(steps) is int:
        _steps = steps
    else:
        raise(Exception(f"{steps} is not int nor list of two valus"))


    if _steps >= 0:
        padd_left = _steps
        padd_right = 0
        _from,_to =  (0,len(x))

    if _steps < 0:
        padd_left = 0
        padd_right = abs(_steps)
        _from,_to =  (padd_right,len(x)+padd_right)

    res = np.pad(x, (padd_left,padd_right), mode='edge')
    res = res[_from:_to]
    
    return res, _steps

def extend_shift(y: Union[np.ndarray, List], x: Union[np.ndarray, List], num:int=30, steps:int=0):
    """extend a signal to `num` samples and shift it by `steps`

    Args:
        y (Union[np.ndarray, List]): signal
        x (Union[np.ndarray, List]): linspace
        num (int, optional): length of the ouput signal. Defaults to 60.
        steps (int, optional): shifting amount. Defaults to 0.

    Returns:
        [type]: [description]
    """
    resp = extend(y,x,num)
    resp,_ = shift(resp, steps)
    return resp