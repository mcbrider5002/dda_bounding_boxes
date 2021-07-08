import itertools
from abc import ABC, abstractmethod
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from grid import GenericBox, LocatorGrid, AllOverlapGrid

#how do we colour?
#fixed set of colours - only works up to fixed n
#by intensity
#by number of parents
#unique colour assigned to each parent then interpolate for overlaps

#by unique colour for parents
#collect set of top-level boxes, give one colour to each, then interpolate for anything that has more than one parent
#if there's no path from one box to another when we build a graph of their overlaps, we can re-use colours
#how do we choose colours?
    #by distinctiveness - maximise distance of n points on the colour cube
    #don't know how to do this analytically yet...
    #so, let's work out how to add points greedily one-at-a-time to maximise some distance function
    #having done that we can create a configuration from a single seed point
    #or optimise an existing configuration by scanning each point for the one that would give the greatest change when replaced with a greedily optimised point, repeat
    #can randomly seed as well
    #we can also plot these points in 3d space to build intuition
    #can rotate soln. to get different colour combo?

class RGBColour():
    def __init__(self, R, G, B): self.R, self.G, self.B = R, G, B
    def __repr__(self): return f"RGBColour({self.R}, {self.G}, {self.B})"
    def squash(self): return (self.R / 255.0, self.G / 255.0, self.B / 255.0)
    def interpolate(self, other, weight=0.5):
        def helper(x, y): return x + (y - x) * weight
        return RGBColour(helper(self.R, other.R), helper(self.G, other.G), helper(self.B, other.B))

class ColourMap():

    PURE_RED = RGBColour(255, 0, 0)
    PURE_GREEN = RGBColour(0, 255, 0)
    PURE_BLUE = RGBColour(0, 0, 255)
    
    RED = RGBColour(237, 28, 36)
    ORANGE = RGBColour(255, 127, 39)
    YELLOW = RGBColour(255, 242, 0)
    GREEN = RGBColour(34, 177, 76)
    LIGHT_BLUE = RGBColour(0, 162, 232)
    INDIGO = RGBColour(63, 72, 204)
    VIOLET = RGBColour(163, 73, 164)

    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def assign_colours(self, boxes, key): pass
    
    def add_to_subplot(self, ax, boxes, key):
        for box, colour in self.assign_colours(boxes, key):
            ax.add_patch(patches.Rectangle((box.pt1.x, box.pt1.y), box.pt2.x - box.pt1.x, box.pt2.y - box.pt1.y, linewidth=1, ec="black", fc=colour.squash()))

    def get_plot(self, boxes, key):
        fig, ax = plt.subplots(1)
        self.add_to_subplot(ax, boxes, key)
        ax.set_xlim([0, max(b.pt2.x for b in boxes)])
        ax.set_ylim([0, max(b.pt2.y for b in boxes)])
        return plt
        
class FixedMap(ColourMap):
    '''Maps discrete values to colours, up to some cap.'''
    def __init__(self, mapping): 
        self.mapping = mapping
        
    def assign_colours(self, boxes, key):
        return ((b, self.mapping[min(key(b), len(self.mapping) - 1)]) for b in boxes)
        
class InterpolationMap(ColourMap):
    '''Assigns colours by interpolating between n colours based on a key value.'''
    def __init__(self, colours):
        self.colours = list(colours)
        
    def assign_colours(self, boxes, key):
        minm = min(key(b) for b in boxes)
        maxm = max(key(b) for b in boxes)
        intervals = [minm + i * (maxm - minm) / (len(self.colours) - 1) for i in range(len(self.colours))]
    
        def get_colour(box):
            i = next(i-1 for i, threshold in enumerate(intervals) if threshold >= key(box))
            weight = (key(box) - intervals[i]) / (intervals[i + 1] - intervals[i])
            return self.colours[i].interpolate(self.colours[i+1], weight=weight)
        
        return ((b, get_colour(b)) for b in boxes)
        
def generate_boxes(n, max_xwidth, max_ywidth):
    inner_x1 = 0.75 * max_xwidth / 2.0
    inner_x2 = max_xwidth - inner_x1
    mid_x1 = inner_x1 / 2
    mid_x2 = max_xwidth - mid_x1
    x_step = inner_x1 / (n - 1) if n > 1 else 0
    
    inner_y1 = 0.75 * max_ywidth / 2.0
    inner_y2 = max_ywidth - inner_y1
    mid_y1 = inner_y1 / 2
    mid_y2 = max_ywidth - mid_y1
    y_step = inner_y1 / (n - 1) if n > 1 else 0
    
    def make_box(i):
        steps = (i + 1) // 2
        sign = 1 if i % 2 == 0 else -1
        x1 = round(mid_x1 - (i / 2 * x_step) * sign)
        x2 = round(mid_x2 + (i / 2 * x_step) * sign)
        y1 = round(mid_y1 + (i / 2 * y_step) * sign)
        y2 = round(mid_y2 - (i / 2 * y_step) * sign)
        return GenericBox(x1, x2, y1, y2)
    return [make_box(i) for i in range(n)]
        
def main():
    matplotlib.use('TKAgg') #qt is broken on one of my systems - this is workaround

    def split_all(grid, boxes):
        for b in boxes: grid.register_box(b)
        return grid.boxes_by_overlaps()
    
    boxes = [
        GenericBox(0, 10, 0, 30, intensity=40), 
        GenericBox(5, 15, 0, 30, intensity=30), 
        GenericBox(0, 10, 15, 45, intensity=20), 
        GenericBox(0, 17, 0, 30, intensity=10)
    ]

    grid = AllOverlapGrid(0, 1440, 100, 0, 1500, 100)
    all_splits = list(itertools.chain(*split_all(grid, boxes)))
    
    cmap = FixedMap([ColourMap.RED, ColourMap.ORANGE, ColourMap.YELLOW, ColourMap.GREEN, ColourMap.LIGHT_BLUE, ColourMap.INDIGO, ColourMap.VIOLET])
    cmap.get_plot(list(reversed(boxes)), key=lambda b: {b : i for i, b in enumerate(reversed(boxes))}[b]).show()
    cmap.get_plot(all_splits, key=lambda b: len(b.top_level_boxes()) - 1).show()
    
    cmap = InterpolationMap([ColourMap.YELLOW, ColourMap.RED])
    cmap.get_plot(all_splits, key=lambda b: b.intensity).show()
    cmap.get_plot(all_splits, key=lambda b: len(b.top_level_boxes())).show()
    
    boxes = [generate_boxes(i, 80, 80) for i in range(1, 11)]
    shifts = ((xshift, yshift) for yshift in range(200, -1, -100) for xshift in range(0, 301, 100))
    for ls, (xshift, yshift) in zip(boxes, shifts): 
        for b in ls: b.shift(xshift=xshift, yshift=yshift)
    grid = AllOverlapGrid(0, 1440, 100, 0, 1500, 100)
    all_splits = list(itertools.chain(*split_all(grid, itertools.chain(*boxes))))
        
    cmap = InterpolationMap([ColourMap.PURE_BLUE, ColourMap.PURE_RED])
    keydict = {b : i for ls in boxes for i, b in enumerate(ls)}
    cmap.get_plot(list(itertools.chain(*boxes)), key=lambda b: keydict[b]).show()
    cmap.get_plot(all_splits, key=lambda b: len(b.top_level_boxes())).show()
    
    def num_boxes(boxes):
        for ls in boxes:
            grid = AllOverlapGrid(0, 1440, 100, 0, 1500, 100)
            yield (len(ls), len(list(itertools.chain(*split_all(grid, ls)))))
            
    def true_box_counts():
        i, n = 0, 0
        while(True):
            if(i < 1): n = 1
            else: n += (2 * i + 1) + (2 * (i-1) + 1)
            i += 1
            yield n
            
    print(", ".join(f"({x} -> {y})" for x, y in num_boxes(boxes)))
    print(", ".join(f"({x} -> {y})" for x, y in zip(range(1, 11), true_box_counts())))
    assert all(x == y for (_, x), y in zip(num_boxes(boxes), true_box_counts())), "True box counts didn't match observed box counts for test example!"

if __name__ == "__main__": main()