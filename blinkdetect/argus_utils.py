#!/usr/bin/env python3
# coding: utf-8

import json
import os
import numpy as np
import pandas as pd
from typing import List
from argusutil.annotation.annotation import AnnotationOfIntervals, Interval,  Unit
from argusutil.annotation import deserializeAnnotation


# VIDEO_IDS = ["35", "37", "38", "40", "42", "47"]

dataset = os.path.join(os.path.dirname(__file__), "..", "dataset", "blinking-gt")
gt_start_end_index_path = os.path.join(dataset,  "gt_start_end_index.csv")

def annotation_for_name(name, annotation_predictions, path):
    return deserializeAnnotation(json.load(open(path.joinpath([annotation_file for annotation_file in annotation_predictions if name in annotation_file][0]), "r")))

def get_blinking_annotation(pid: str):

    # assert pid.strip() in VIDEO_IDS, f"participant {pid} not available"

    gt_blinking_annotation = deserializeAnnotation(json.load(
            open(os.path.join(dataset,"gt_index", f"aniko_annot_{pid}_blinking.json"), "r")))
    
    gt_start_end = pd.read_csv(gt_start_end_index_path)

    gt_start_v = gt_start_end[gt_start_end["subject"] == int(pid)]["gt_start_index"].iat[0]
    gt_stop_v = gt_start_end[gt_start_end["subject"] == int(pid)]["gt_stop_index"].iat[0]

    return gt_blinking_annotation.slice(gt_start_v, gt_stop_v, sliceIntervalsOnEdge=True, shiftToZero=True)

def get_intervals(binary_signal: List, start_idx:int=0, val:int=0):
    """get intervals from a singal that correspond to the consecutive same value `val`

    Args:
        binary_signal (List): [description]
        start_idx (int, optional): [description]. Defaults to 1.
        val (int, optional): [description]. Defaults to 0.
    """
    start = None
    intervals = AnnotationOfIntervals(Unit.INDEX)
    for idx, _val in enumerate(binary_signal, start_idx):
        if _val == val:
            if start is None:
                start = idx
            end = idx
        else:
            if start is not None and end > start:
                intervals.add(Interval(start,end))
            start = None
        if idx == (len(binary_signal)-start_idx - 1) and start is not None and end > start:
            intervals.add(Interval(start,end))
    return intervals

def get_intervals_between(binary_signal: List, start_idx:int=0, val:int=0):
    """get intervals from a signal that correspond to the consecutive values in a range [-abs(val),abs(val)]

    Args:
        binary_signal (List): [description]
        start_idx (int, optional): [description]. Defaults to 1.
        val (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    start = None
    val = abs(val)
    intervals = AnnotationOfIntervals(Unit.INDEX)

    for idx, _val in enumerate(binary_signal, start_idx):
        if _val>-val and _val<val:
            
            if start is None:
                start = idx
            end = idx
        else:
            if start is not None and end > start:
                intervals.add(Interval(start,end))
            start = None
        if idx == (len(binary_signal)-start_idx-1) and start is not None and end > start:
            intervals.add(Interval(start,end))
    return intervals


if __name__ == "__main__":
    res = get_blinking_annotation("35")
    # print(get_intervals([0,0,0,0], val=0))
    sig = [-30,32.902053780795086, 33.07398840456521, 34.99646986075648, 35.65296884615576, 34.15734797951384, 34.865442491304194, 37.06310542044604, 39.50996022809948, 36.36872397530099, 39.937563702510104, 38.37214456032004, 39.85691327760786, 37.98714895503762, 36.63564086639466, 35.9476769993411, 36.28513224833076, 36.697269006499454, 33.25366333786448, 32.25707496938849, 31.73459691215714, 30.232698134141202, 31.591232492057326, 31.8139154839889, 30.939161177284962, 31.080864898304416, 30.96851026275865, 30.87768871119526, 29.78008246501561, 31.271437828458755, 30.420877477774138, 29.183159594239527, 30.156254700888706, 29.45878725652404, 29.31115909487054, 27.17675626460166, 28.007315038758584, 27.10755077223243, 27.23142412197663, 27.799241745400575, 27.187021962885723, 27.456189690428136, 27.447270954860873, 27.73461604316255, 28.189643450722457, 27.05903497668478, 27.137917162880132, 28.13647869802834, 28.88185878760167, 29.043709375929698, 29.09665158492682, 28.433241575082636, 28.206220592142518, 27.244956033603223, 27.786936572467795, 27.505899606354188, 27.15039187410215, 27.44679954230337, 27.480702130538596, 27.56797469881198, 28.21885667440842, 27.481058204431267, 27.220215352127845, 27.094724081008852, 27.151929002268037, 26.30678640921252, 27.47680079089852]
    print(get_intervals_between(binary_signal=sig, val=34))

    # print(res[0])
    # for interval in AnnotationOfIntervals(Unit.INDEX, [Interval(1,2), Interval(4,5)]):
    #     print(interval)
    #     print(interval.start, interval.stop)