#!/usr/bin/env python3
# coding: utf-8

from blinkdetect.signal_1d import prolong, extend
import numpy as np

def resample_noblink(y_in, start, stop, samples=None, upsampling=True):
    y = y_in.copy()
    y_roi = y[start:stop+1].copy()
    # blink_length = stop-start+1
    #     
    # x = np.linspace(0,len(y)-1, len(y))
    x_roi = np.linspace(start,stop, len(y_roi))

    blink = np.zeros(len(y)).tolist()
    # blink[start:stop+1] = np.ones(blink_length, dtype=np.int32).tolist()
    blink_roi = blink[start:stop+1].copy()
    # prolong
    min_len=min(6, len(y_roi))
    max_len=max(21,len(y_roi))
    #
    y_sampled, _num = prolong(y_roi, x_roi, upsampling=upsampling, min_len=min_len, max_len=max_len, Num=samples)
    _x_sampled = np.linspace(start,start+len(y_sampled)-1, len(y_sampled))
    try:
        blink_sampled, _num = prolong(blink_roi,x_roi, min_len=min_len, max_len=max_len,Num=_num)
    except Exception as e:
        print()
        print(len(blink_roi),len(x_roi),_num, start, stop)
        print()
        raise(e)


    # extend
    # blink period
    _blink_sampled = blink.copy()
    _blink_sampled[start:stop+1] = blink_sampled.copy()
    # signal
    _y_sampled = y.copy()
    _y_sampled[start:stop+1] = y_sampled.copy()
    # 
    _y_extended, _x_extended = extend(_y_sampled,_x_sampled)
    _blink_extended, _ = extend(_blink_sampled,_x_sampled, val=0)
    return _y_extended, _blink_extended.astype(np.int32), _num

def resample_blink(y_in, start, stop, samples=None, upsampling=True):
    y = y_in.copy()
    y_roi = y[start:stop+1].copy()
    blink_length = stop-start+1
    #     
    # x = np.linspace(0,len(y)-1, len(y))
    x_roi = np.linspace(start,stop, len(y_roi))

    blink = np.zeros(len(y)).tolist()
    blink[start:stop+1] = np.ones(blink_length, dtype=np.int32).tolist()
    blink_roi = blink[start:stop+1].copy()
    # prolong
    min_len=min(6, len(y_roi))
    max_len=max(21,len(y_roi))
    #
    y_sampled, _num = prolong(y_roi, x_roi, upsampling=upsampling, min_len=min_len, max_len=max_len, Num=samples)
    _x_sampled = np.linspace(start,start+len(y_sampled)-1, len(y_sampled))
    
    blink_sampled, _num = prolong(blink_roi,x_roi, min_len=min_len, max_len=max_len,Num=_num)


    # extend
    # blink period
    _blink_sampled = blink.copy()
    _blink_sampled[start:stop+1] = blink_sampled.copy()
    # signal
    _y_sampled = y.copy()
    _y_sampled[start:stop+1] = y_sampled.copy()
    # signals returned of length 60
    _y_extended, _x_extended = extend(_y_sampled,_x_sampled, num=60)
    _blink_extended, _ = extend(_blink_sampled,_x_sampled, num=60,val=0)
    return _y_extended, _blink_extended.astype(np.int32), _num

def upsample_blink(y_in, start, stop, samples=None):
    return resample_blink(y_in, start, stop,samples, True)

def downsample_blink(y_in, start, stop, samples=None):
    return resample_blink(y_in, start, stop,samples, False)