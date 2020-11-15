import math
import random
import itertools
import numpy as np
from decimal import Decimal
from collections import defaultdict
from abc import ABC, abstractmethod

'''
We want to:
Generate random bounding boxes (multiple injections)
Fit grid onto injections
Find overlap of bounding boxes
(Plot results for reasons of intuition? Use matplotlib rectangles and (potentially) text to indicate box area...)
'''

#TODO:
#implement exact method
#@dataclass?
#plot boxes?

#add bounding box points to RoI class
#priorisation function based on peak-picking and non-overlap
#controller with TopN for first run, intensity*non-overlap on further runs
#m/z grid doesn't have to be less than half of mass tolerance
#rt grid should be a bit less than the sampling time

class Point():
    def __init__(self, x, y):
        self.x, self.y = x, y

class Box():
    def __init__(self, x1, x2, y1, y2):
        self.pt1 = Point(min(x1, x2), min(y1, y2))
        self.pt2 = Point(max(x1, x2), max(y1, y2))
        
    def __repr__(self): return "Box({} {})".format(self.pt1, self.pt2)
    def area(self): return (x2 - x1) * (y2 - y1)
        
class GenericBox(Box):
    '''Makes no particular assumptions about bounding boxes.'''
    
    def __repr__(self): return "Generic{}".format(super().__repr__(self))
    
    def overlaps_with_box(self, other_box):
        return (self.pt1.x < other_box.pt2.x and self.pt2.x > other_box.pt1.x) and (self.pt1.y < other_box.pt2.y and self.pt2.y > other_box.pt1.y)
    
    def contains_box(self, other_box):
        return (
                self.pt1.x <= other_box.pt1.x 
                and self.pt1.y <= other_box.pt1.y 
                and self.pt2.x >= other_box.pt2.x 
                and self.pt2.y >= other_box.pt2.y
               )
               
    def split_box(self, other_box):
        if(not self.overlaps_with_box(other_box)): return None
        x1, x2, y1, y2 = self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y
        split_boxes = []
        if(other_box.pt1.x > self.pt1.x):
            x1 = other_box.pt1.x
            split_boxes.append(GenericBox(self.pt1.x, x1, y1, y2))
        if(other_box.pt2.x > self.pt2.x):
            x2 = other_box.pt2.x
            split_boxes.append(GenericBox(x2, self.pt2.x, y1, y2))
        if(other_box.pt1.y > self.pt1.y):
            y1 = other_box.pt1.y
            split_boxes.append(GenericBox(x1, x2, self.pt1.y, y1))
        if(other_box.pt2.y > self.pt2.y):
            y2 = other_box.pt2.y
            split_boxes.append(GenericBox(x1, x2, y2, self.pt2.y))
        if(len(split_boxes) <= 2):
            self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y = x1, x2, y1, y2
            return None
        return split_boxes
            
class SpecialBox(Box):
    '''Assumes all boxes have an edge at y=0 and that opposite edge is at y>0.'''

    def __repr__(self): return "Special{}".format(super().__repr__(self))

    def overlaps_with_box(self, other_box): 
        return (self.pt1.x < other_box.pt2.x and self.pt2.x > other_box.pt1.x)

    def contains_box(self, other_box):
        return (
                self.pt1.x <= other_box.pt1.x 
                and self.pt2.x >= other_box.pt2.x 
                and self.pt2.y >= other_box.pt2.y
               )
               
    def split_box(self, other_box):
        pass

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
        rt_box_range = (int(box.pt1.x / self.rt_box_size), int(box.pt2.x / self.rt_box_size) + 1)
        mz_box_range = (int(box.pt1.y / self.mz_box_size), int(box.pt2.y / self.mz_box_size) + 1)
        total_boxes = (rt_box_range[1] - rt_box_range[0]) * (mz_box_range[1] - mz_box_range[0])
        return rt_box_range, mz_box_range, total_boxes
        
    @abstractmethod
    def box_non_overlap(self, box, *boxes): pass
        
class DictGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return defaultdict(list)
    
    def box_non_overlap(self, box, *boxes):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        def non_overlap(rt, mz):
            result = 1.0 if not self.boxes[(rt, mz)] else 0.0
            self.boxes[(rt, mz)].append(box)
            return result
        return sum(non_overlap(rt, mz) for rt in range(*rt_box_range) for mz in range(*mz_box_range)) / total_boxes
    
class ArrayGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return np.array([[False for mz in mzboxes] for rt in rtboxes])
    
    def box_non_overlap(self, box, *boxes):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        boxes = self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]
        falses = total_boxes - np.sum(boxes)
        self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]] = True
        return falses / total_boxes

class BoxEnv():

    def __init__(self, min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.max_mz = max_mz
        self.min_x1, self.max_x1 = min_rt, max_rt - max_xlen
        self.min_xlen, self.max_xlen, self.min_ylen, self.max_ylen = min_xlen, max_xlen, min_ylen, max_ylen
        self.injection_index = 0
        self.boxes_by_injection = [[]]
        self.grid = None
        
    def set_injection_no(n): self.injection_no = n
    def next_injection(n):
        self.boxes_by_injection.append([])
        self.injection_index += 1
        
    def init_grid(self, grid_class, rt_box_size, mz_box_size):
        self.grid = grid_class(self.min_rt, self.max_rt, rt_box_size, 0, self.max_mz, mz_box_size)
        
    def generate_box(self):
        x1 = random.uniform(self.min_x1, self.max_x1)
        xlen = random.uniform(self.min_xlen, self.max_xlen)
        ylen = random.uniform(self.min_ylen, self.max_ylen)
        return Box(x1, x1 + xlen, 0, ylen)

    @classmethod
    def random_boxenv(cls, no_injections=3):
        min_rt, max_rt = 0, random.randint(1000, 2000)
        max_mz = random.randint(1000, 3000)
        min_xlen = random.randint(1, 25)
        max_xlen = random.randint(min_xlen, 50)
        min_ylen = random.randint(100, 1000)
        max_ylen = max_mz - min_ylen
        boxenv = BoxEnv(min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen)
        for i in range(no_injections): boxenv.boxes_by_injection.append([boxenv.generate_box() for j in range(50)])
        return boxenv
        
    @staticmethod
    def dummy_non_overlap(box, *other_boxes): 
        return 1.0
        
    def grid_non_overlap(self, box, *other_boxes):
        return self.grid.box_non_overlap(box, *other_boxes)
    
    @staticmethod
    def splitting_non_overlap(box, *other_boxes):
        new_boxes = [box] #more efficient splitting with set?
        for b in other_boxes: #potentially filter boxes down via grid with large boxes for this loop?
            if(box.overlaps_with_box(b)): #quickly exits any box not overlapping new box
                for b2 in new_boxes:
                    if(b.contains_box(b2)):
                        new_boxes.remove(b2)
                        if(not new_boxes): return 0
                    else:
                        split_boxes = b2.split_boxes(b)
                        if(not split_boxes is None):
                            new_boxes.remove(b2)
                            new_boxes.extend(split_boxes)
        return sum(b.area() for b in new_boxes)
        
        '''case analysis:
            - your box is contained within a previous box, in which case area is 0 (remove box from list, return if list is empty)
            - your box contains a previous box, in which case split your box into up to four new boxes, one for each edge they don't have in common (if they have all four edges in common, then box is completely overlapped)
            - neither of the previous statements are true, but boxes still overlap, then new box can contain 0, 1 or 2 of previous box's corners
              - if 0, then previous box completely overlaps one dimension, and you can just crop the new box
              - if 1, there will be a corner inside your new box, and you will need to split it into 2
              - if 2, box overlaps part of your dimension, so you will need to split into 3
        ^could be simplified by organising by number of points
        
        boxes could be potentially sorted by size (O(n) insert time in worst-case)'''
        
    def box_uniqueness_by_injection(self, non_overlap_f):
        return [[non_overlap_f(box, *self.boxes_by_injection[:i], *inj[:j]) for j, box in enumerate(inj)] for i, inj in enumerate(self.boxes_by_injection)]
        
    class BoxScores():
        def __init__(non_overlap):
            self.non_overlap = non_overlap
        
    def add_boxes(non_overlap_f, *boxes):
        scores = []
        for b in boxes:
            non_overlap = non_overlap_f(b, itertools.chain(*self.boxes_by_injection))
            scores.append(BoxScores(non_overlap))
            self.boxes_by_injection[self.injection_index].append(b)
        return scores
        
def main():
    def run_area_calcs(boxenv, rt_box_size, mz_box_size):
        print("Run area calcs start!")
        boxenv.init_grid(DictGrid, rt_box_size, mz_box_size)
        scores_by_injection = boxenv.box_uniqueness_by_injection(boxenv.grid_non_overlap)
        print(scores_by_injection)
    
        boxenv.init_grid(ArrayGrid, rt_box_size, mz_box_size)
        scores_by_injection_2 = boxenv.box_uniqueness_by_injection(boxenv.grid_non_overlap)
        print(scores_by_injection_2)
    
    boxenv = BoxEnv.random_boxenv()
    scores_by_injection = boxenv.box_uniqueness_by_injection(boxenv.dummy_non_overlap)
    print(scores_by_injection)
    run_area_calcs(boxenv, (boxenv.max_rt - boxenv.min_rt) / 1000, boxenv.max_mz / 1000)
    
    boxenv = BoxEnv(0, 50, 50, 2, 3, 2, 3)
    boxenv.boxes_by_injection = [[Box(0, 10, 0, 30), Box(5, 15, 0, 30), Box(0, 10, 15, 45), Box(0, 17, 0, 30)]]
    run_area_calcs(boxenv, 0.2, 0.2)
    
main()