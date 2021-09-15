import os
import numpy as np
from argusutil.annotation.annotation import AnnotationOfIntervals, Interval, Unit

def flip_signal(y):
    _mean = np.mean(y)
    inv_data_y = (y-_mean)*(-1) + _mean
    return inv_data_y


# https://www.kaggle.com/hakkoz/eye-blink-detection-1-simple-model?scriptVersionId=33335937&cellId=26
# read tag file and construct "closeness_list" and "blinks_list"
def read_annotations_tag(input_file: str):
    """read annotations by blinkmatters.com
    """
    name, ext = os.path.splitext(input_file)
    assert ext==".tag", "file extension is not .tag"
    # define variables 
    blink_start = 1
    blink_end = 1
    blink_info = (0,0)
    blink_list = []
    closeness_list = []

    # Using readlines() 
    file1 = open(input_file, "r", encoding="utf-8") 
    Lines = file1.readlines() 

    # find "#start" line 
    start_line = 1
    for line in Lines: 
        clean_line=line.strip()
        if clean_line=="#start":
            break
        start_line += 1

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    for index in range(len(Lines[start_line : -1])): # -1 since last line will be"#end"
        
        # read previous annotation and current annotation 
        prev_annotation=Lines[start_line+index-1].split(':')
        current_annotation=Lines[start_line+index].split(':')
        
        # if previous annotation is not "#start" line and not "blink" and current annotation is a "blink"
        if prev_annotation[0] != "#start\n" and prev_annotation[1] == "-1" and int(current_annotation[1]) > 0:
            # it means a new blink starts so save frame id as starting frame of the blink
            blink_start = int(current_annotation[0])
        
        # if previous annotation is not "#start" line and is a "blink" and current annotation is not a "blink"
        if prev_annotation[0] != "#start\n" and int(prev_annotation[1]) > 0 and current_annotation[1] == "-1":
            # it means a new blink ends so save (frame id - 1) as ending frame of the blink
            blink_end = int(current_annotation[0]) - 1
            # and construct a "blink_info" tuple to append the "blink_list"
            blink_info = Interval(blink_start,blink_end)
            blink_list.append(blink_info)
        
        # if current annotation consist fully closed eyes, append it also to "closeness_list" 
        if current_annotation[3] == "C" and current_annotation[5] == "C":
            closeness_list.append(1)
        
        else:
            closeness_list.append(0)
    
    file1.close()
    blinks_intervals = AnnotationOfIntervals(Unit.INDEX, blink_list)
    return closeness_list, blinks_intervals
    
    
    
    