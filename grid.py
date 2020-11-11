import math
import random
import itertools
import numpy as np
from decimal import Decimal
from abc import ABC, abstractmethod

'''
We want to:
Generate random bounding boxes (multiple scans)
Fit grid onto scans
Find overlap of bounding boxes
(Plot results for reasons of intuition? Use matplotlib rectangles and (potentially) text to indicate box area...)
'''

class Box():
    def __init__(self, x1, x2, y1, y2):
        self.pt1 = min(x1, x2), min(y1, y2)
        self.pt2 = max(x1, x2), max(y1, y2)
        
    def __repr__(self): return "Box({} {})".format(self.pt1, self.pt2)
    def area(self): return (x2 - x1) * (y2 - y1)

class Grid():

    @staticmethod
    @abstractmethod
    def init_boxes(): pass

    def __init__(self, min_rt, max_rt, rt_box_size, min_mz, max_mz, mz_box_size):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.min_mz, self.max_mz = min_mz, max_mz
        self.rt_box_size, self.mz_box_size = rt_box_size, mz_box_size
        self.box_area = float(Decimal(rt_box_size) * Decimal(mz_box_size))
        
        rtboxes = range(0, int((self.max_rt - self.min_rt) / rt_box_size) + 1)
        mzboxes = range(0, int((self.max_mz - self.min_mz) / mz_box_size) + 1)
        self.boxes = self.init_boxes(rtboxes, mzboxes)
        
    def get_box_ranges(self, box):
        rt_box_range = (int(box.pt1[0] / self.rt_box_size), int(box.pt2[0] / self.rt_box_size) + 1)
        mz_box_range = (int(box.pt1[1] / self.mz_box_size), int(box.pt2[1] / self.mz_box_size) + 1)
        return rt_box_range, mz_box_range
        
    @abstractmethod
    def box_non_overlap(self, box, *boxes): pass
        
class DictGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return {(rt, mz) : [] for rt in rtboxes for mz in mzboxes}
    
    def box_non_overlap(self, box, *boxes):
        rt_box_range, mz_box_range = self.get_box_ranges(box)
        def non_overlap(rt, mz):
            result = self.box_area if not self.boxes[(rt, mz)] else 0.0
            self.boxes[(rt, mz)].append(box)
            return result
        return sum(non_overlap(rt, mz) for rt in range(*rt_box_range) for mz in range(*mz_box_range))
    
class ArrayGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return np.array([[False for mz in mzboxes] for rt in rtboxes])
    
    def box_non_overlap(self, box, *boxes):
        rt_box_range, mz_box_range = self.get_box_ranges(box)
        boxes = self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]
        result = boxes.shape[0] * boxes.shape[1] - np.sum(boxes)
        self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]] = True
        return result * self.box_area

class BoxEnv():

    def __init__(self, min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.max_mz = max_mz
        self.min_x1, self.max_x1 = min_rt, max_rt - max_xlen
        self.min_xlen, self.max_xlen, self.min_ylen, self.max_ylen = min_xlen, max_xlen, min_ylen, max_ylen
        self.boxes_by_scan = []
        self.grid = None
        
    def init_grid(self, grid_class, rt_box_size, mz_box_size):
        self.grid = grid_class(self.min_rt, self.max_rt, rt_box_size, 0, self.max_mz, mz_box_size)
        
    def generate_box(self):
        x1 = random.uniform(self.min_x1, self.max_x1)
        xlen = random.uniform(self.min_xlen, self.max_xlen)
        ylen = random.uniform(self.min_ylen, self.max_ylen)
        return Box(x1, x1 + xlen, 0, ylen)

    @classmethod
    def random_boxenv(cls, no_scans=3):
        min_rt, max_rt = 0, random.randint(1000, 2000)
        max_mz = random.randint(1000, 3000)
        min_xlen = random.randint(1, 25)
        max_xlen = random.randint(min_xlen, 50)
        min_ylen = random.randint(100, 1000)
        max_ylen = max_mz - min_ylen
        boxenv = BoxEnv(min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen)
        for i in range(no_scans): boxenv.boxes_by_scan.append([boxenv.generate_box() for i in range(50)])
        return boxenv
        
    @staticmethod
    def dummy_non_overlap(box, *other_boxes): 
        return 1.0
        
    def grid_non_overlap(self, box, *other_boxes):
        return self.grid.box_non_overlap(box, *other_boxes)
    
    @staticmethod
    def splitting_non_overlap(box, *other_boxes):
        '''
        new_boxes = [box]
        for b in other_boxes: #potentially filter boxes down via grid with large boxes for this loop?
            if(overlap(box, b)): #quickly exits any box not overlapping new box
                for b2 in new_boxes:
                    #case analysis
        return sum(b.area() for b in new_boxes)
        
        case analysis:
            - your box is contained within a previous box, in which case area is 0 (remove box from list, return if list is empty)
            - your box contains a previous box, in which case split your box into up to four new boxes, one for each edge they don't have in common (if they have all four edges in common, then box is completely overlapped)
            - neither of the previous statements are true, but boxes still overlap, then new box can contain 0, 1 or 2 of previous box's corners
              - if 0, then previous box completely overlaps one dimension, and you can just crop the new box
              - if 1, there will be a corner inside your new box, and you will need to split it into 2
              - if 2, box overlaps part of your dimension, so you will need to split into 3
        ^could be simplified by organising by number of points
        
        boxes could be potentially sorted by size (O(n) insert time in worst-case)
        '''
        pass
        
    def box_uniqueness_by_scan(self, non_overlap):
        return [[non_overlap(box, *self.boxes_by_scan[:i], *scan[:j]) for j, box in enumerate(scan)] for i, scan in enumerate(self.boxes_by_scan)]
        
def main():
    def run_area_calcs(boxenv, rt_box_size, mz_box_size):
        boxenv.init_grid(DictGrid, rt_box_size, mz_box_size)
        scores_by_scan = boxenv.box_uniqueness_by_scan(boxenv.grid_non_overlap)
        print(scores_by_scan)
    
        boxenv.init_grid(ArrayGrid, rt_box_size, mz_box_size)
        scores_by_scan_2 = boxenv.box_uniqueness_by_scan(boxenv.grid_non_overlap)
        print(scores_by_scan_2)
    
    boxenv = BoxEnv.random_boxenv()
    scores_by_scan = boxenv.box_uniqueness_by_scan(boxenv.dummy_non_overlap)
    print(scores_by_scan)
    run_area_calcs(boxenv, (boxenv.max_rt - boxenv.min_rt) / 1000, boxenv.max_mz / 1000)
    
    boxenv = BoxEnv(0, 50, 50, 2, 3, 2, 3)
    boxenv.boxes_by_scan = [[Box(0, 10, 0, 30), Box(5, 15, 0, 30), Box(0, 10, 15, 45), Box(13, 17, 0, 30)]]
    run_area_calcs(boxenv, 0.05, 0.05)
    
main()