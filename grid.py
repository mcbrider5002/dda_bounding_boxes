import math
import random
import itertools
import numpy as np
from decimal import Decimal
from collections import defaultdict
from abc import ABC, abstractmethod
from time import perf_counter

'''
We want to:
Generate random bounding boxes (multiple injections)
Fit grid onto injections
Find overlap of bounding boxes
(Plot results for reasons of intuition? Use matplotlib rectangles and (potentially) text to indicate box area...)
'''

#TODO:
#use scoring object?
#implement general case of box splitting?
#@dataclass?
#plot boxes?

#add bounding box points to RoI class
#priorisation function based on peak-picking and non-overlap
#controller with TopN for first run, intensity*non-overlap on further runs
#m/z grid doesn't have to be less than half of mass tolerance
#rt grid should be a bit less than the sampling time

class Point():
    def __init__(self, x, y): self.x, self.y = float(x), float(y)
    def __repr__(self): return "Point({}, {})".format(self.x, self.y)

class Box():
    def __init__(self, x1, x2, y1, y2):
        self.pt1 = Point(min(x1, x2), min(y1, y2))
        self.pt2 = Point(max(x1, x2), max(y1, y2))
        
    def __repr__(self): return "Box({}, {})".format(self.pt1, self.pt2)
    def __hash__(self): return (self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y).__hash__()
    def area(self): return (self.pt2.x - self.pt1.x) * (self.pt2.y - self.pt1.y)
    def copy(self): return type(self)(self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y)
        
class GenericBox(Box):
    '''Makes no particular assumptions about bounding boxes.'''
    
    def __repr__(self): return "Generic{}".format(super().__repr__())
    
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
        '''Finds 1 to 4 boxes describing the polygon of area of this box not overlapped by other_box.
           If one box is found, crops this box to dimensions of that box, and returns None.
           Otherwise, returns list of 2 to 4 boxes. Number of boxes found is equal to number of edges overlapping area does NOT share with this box.'''
        if(not self.overlaps_with_box(other_box)): return None
        x1, x2, y1, y2 = self.pt1.x, self.pt2.x, self.pt1.y, self.pt2.y
        split_boxes = []
        if(other_box.pt1.x > self.pt1.x):
            x1 = other_box.pt1.x
            split_boxes.append(GenericBox(self.pt1.x, x1, y1, y2))
        if(other_box.pt2.x < self.pt2.x):
            x2 = other_box.pt2.x
            split_boxes.append(GenericBox(x2, self.pt2.x, y1, y2))
        if(other_box.pt1.y > self.pt1.y):
            y1 = other_box.pt1.y
            split_boxes.append(GenericBox(x1, x2, self.pt1.y, y1))
        if(other_box.pt2.y < self.pt2.y):
            y2 = other_box.pt2.y
            split_boxes.append(GenericBox(x1, x2, y2, self.pt2.y))
        if(len(split_boxes) == 1):
            self.pt1, self.pt2 = split_boxes[0].pt1, split_boxes[0].pt2
            return None
        return split_boxes

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
    def non_overlap(self, box): pass
        
    @abstractmethod
    def register_box(self, box): pass   
        
class DictGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return defaultdict(list)
    
    def non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        return sum(float(not self.boxes[(rt, mz)]) for rt in range(*rt_box_range) for mz in range(*mz_box_range)) / total_boxes
        
    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for rt in range(*rt_box_range):
            for mz in range(*mz_box_range): self.boxes[(rt, mz)].append(box)
    
class ArrayGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes): return np.array([[False for mz in mzboxes] for rt in rtboxes])
    
    def non_overlap(self, box):
        rt_box_range, mz_box_range, total_boxes = self.get_box_ranges(box)
        boxes = self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]] 
        return (total_boxes - np.sum(boxes)) / total_boxes
        
    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]] = True
        
class LocatorGrid(Grid):
    @staticmethod
    def init_boxes(rtboxes, mzboxes):
        arr = np.empty((max(rtboxes), max(mzboxes)), dtype=object)
        for i, row in enumerate(arr):
            for j, _ in enumerate(row): arr[i, j] = list() 
        return arr
    
    def get_boxes(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        boxes = []
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for ls in row: boxes.append(ls)
        return boxes
        
    @staticmethod
    def dummy_non_overlap(box, *other_boxes): return 1.0   
        
    @staticmethod
    def splitting_non_overlap(box, *other_boxes):
        new_boxes = [box.copy()]
        for b in other_boxes: #filter boxes down via grid with large boxes for this loop + boxes could be potentially sorted by size (O(n) insert time in worst-case)?
            if(box.overlaps_with_box(b)): #quickly exits any box not overlapping new box
                for b2 in new_boxes:
                    if(b.contains_box(b2)): #your box is contained within a previous box, in which case area is 0 (remove box from list, return if list is empty)
                        new_boxes.remove(b2)
                        if(not new_boxes): return 0
                    else:
                        split_boxes = b2.split_box(b)
                        if(not split_boxes is None):
                            new_boxes.remove(b2)
                            new_boxes.extend(split_boxes)
        return sum(b.area() for b in new_boxes) / box.area()
        
    def non_overlap(self, box):
        return self.splitting_non_overlap(box, *itertools.chain(*self.get_boxes(box)))

    def register_box(self, box):
        rt_box_range, mz_box_range, _ = self.get_box_ranges(box)
        for row in self.boxes[rt_box_range[0]:rt_box_range[1], mz_box_range[0]:mz_box_range[1]]:
            for ls in row: ls.append(box)
        
class BoxEnv():
    def __init__(self, min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen):
        self.min_rt, self.max_rt = min_rt, max_rt
        self.max_mz = max_mz
        self.min_x1, self.max_x1 = min_rt, max_rt - max_xlen
        self.min_xlen, self.max_xlen, self.min_ylen, self.max_ylen = min_xlen, max_xlen, min_ylen, max_ylen
        self.grid = None        
        
    def init_grid(self, grid_class, rt_box_size, mz_box_size):
        self.grid = grid_class(self.min_rt, self.max_rt, rt_box_size, 0, self.max_mz, mz_box_size)
        
    def generate_box(self):
        x1 = random.uniform(self.min_x1, self.max_x1)
        y1 = random.uniform(0, self.max_mz-self.max_ylen)
        xlen = random.uniform(self.min_xlen, self.max_xlen)
        ylen = random.uniform(self.min_ylen, self.max_ylen)
        return GenericBox(x1, x1 + xlen, y1, y1 + ylen)
        
    @classmethod
    def random_boxenv(cls):
        min_rt, max_rt = 0, random.randint(1000, 2000)
        max_mz = random.randint(1000, 3000)
        min_xlen = random.randint(1, 4)
        max_xlen = random.randint(min_xlen, 10)
        min_ylen = random.randint(1, 5)
        max_ylen = random.randint(min_ylen, 10)
        return BoxEnv(min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen)        
        
    def box_score(self, box): return self.grid.non_overlap(box)
    def register_box(self, box): self.grid.register_box(box)
        
    class BoxScores():
        '''Unused for now: use to package other kinds of score later?'''
        def __init__(non_overlap):
            self.non_overlap = non_overlap
        
class TestEnv(BoxEnv):
    def __init__(self, min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen):
        super().__init__(min_rt, max_rt, max_mz, min_xlen, max_xlen, min_ylen, max_ylen)
        self.boxes_by_injection = [[]]

    @classmethod
    def random_boxenv(cls, boxes_per_injection, no_injections):
        boxenv = super().random_boxenv()
        boxenv = TestEnv(boxenv.min_rt, boxenv.max_rt, boxenv.max_mz, boxenv.min_xlen, boxenv.max_xlen, boxenv.min_ylen, boxenv.max_ylen)
        boxenv.boxes_by_injection = [[boxenv.generate_box() for j in range(boxes_per_injection)] for i in range(no_injections)]
        return boxenv
    
    def test_simple_splitter(self):
        return [[LocatorGrid.splitting_non_overlap(box, *itertools.chain(*self.boxes_by_injection[:i]), *inj[:j]) for j, box in enumerate(inj)] for i, inj in enumerate(self.boxes_by_injection)]

    def test_non_overlap(self, grid_class, rt_box_size, mz_box_size):
        self.init_grid(grid_class, rt_box_size, mz_box_size)
        def score_box(box):
            score = self.grid.non_overlap(box)
            self.grid.register_box(box)
            return score
        return [[score_box(b) for b in inj] for inj in self.boxes_by_injection]
        
def main():
    class Timer():
        def __init__(self): self.time = None
        def start_time(self): self.time = perf_counter()
        def end_time(self): return perf_counter() - self.time
        def time_f(self, f):
            self.start_time()
            result = f()
            return result, self.end_time()
    
    def run_area_calcs(boxenv, rt_box_size, mz_box_size):
        def pretty_print(scores):
            print({i : x for i, x in enumerate(itertools.chain(*scores))})
        print("\nRun area calcs start!")
        print("\nDictGrid Scores:")
        scores_by_injection, dict_time = Timer().time_f(lambda: boxenv.test_non_overlap(DictGrid, rt_box_size, mz_box_size))
        pretty_print(scores_by_injection)
    
        print("\nBoolArrayGrid Scores:")
        scores_by_injection_2, array_time = Timer().time_f(lambda: boxenv.test_non_overlap(ArrayGrid, rt_box_size, mz_box_size))
        pretty_print(scores_by_injection_2)
        
        print("\nExact Scores:")
        scores_by_injection_3, exact_time = Timer().time_f(lambda: boxenv.test_simple_splitter())
        pretty_print(scores_by_injection_3)
        
        print("\nExact Scores Grid:")
        rt_box_size, mz_box_size = (boxenv.max_rt - boxenv.min_rt) / 50, boxenv.max_mz / 50
        scores_by_injection_4, exact_grid_time = Timer().time_f(lambda: boxenv.test_non_overlap(LocatorGrid, rt_box_size, mz_box_size))
        pretty_print(scores_by_injection_4)
        
        print("\nDictGrid Time Taken: {}".format(dict_time))
        print("BoolArray Time Taken: {}".format(array_time))
        print("BoxSplitting Time Taken: {}".format(exact_time))
        print("BoxSplitting with Grid Time Taken {}".format(exact_grid_time))
    
    boxenv = TestEnv.random_boxenv(2000, 3)
    run_area_calcs(boxenv, (boxenv.max_rt - boxenv.min_rt) / 10000, boxenv.max_mz / 10000)
    
    boxenv = TestEnv(0, 50, 50, 2, 3, 2, 3)
    boxenv.boxes_by_injection = [[GenericBox(0, 10, 0, 30), GenericBox(5, 15, 0, 30), GenericBox(0, 10, 15, 45), GenericBox(0, 17, 0, 30)]]
    run_area_calcs(boxenv, 0.2, 0.2)
    
main()