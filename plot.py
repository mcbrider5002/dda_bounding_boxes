import itertools
from abc import ABC, abstractmethod
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from grid import GenericBox, LocatorGrid

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

def rgb_squasher(fst, snd, trd): return (fst / 255.0, snd / 255.0, trd / 255.0)
    
class ColourMap():

    PURE_RED = rgb_squasher(255, 0, 0)
    PURE_GREEN = rgb_squasher(0, 255, 0)
    PURE_BLUE = rgb_squasher(0, 0, 255)
    
    RED = rgb_squasher(237, 28, 36)
    ORANGE = rgb_squasher(255, 127, 39)
    YELLOW = rgb_squasher(255, 242, 0)
    GREEN = rgb_squasher(34, 177, 76)
    LIGHT_BLUE = rgb_squasher(0, 162, 232)
    INDIGO = rgb_squasher(63, 72, 204)
    VIOLET = rgb_squasher(163, 73, 164)

    '''Used to assign colours when drawing boxes, by mapping no_overlaps -> colour.'''
    @abstractmethod
    def __init__(self): pass
    
    @abstractmethod
    def get_colour(self): pass

class FixedMap(ColourMap):
    '''Assigns colours using some given fixed mapping -> all non-overlaps exceeding the highest colour value will use it instead.'''
    def __init__(self, mapping): self.mapping = mapping
    def get_colour(self, n): return self.mapping[-1] if n >= len(self.mapping) else self.mapping[n]
    
class InterpolationMap(ColourMap):
    '''Assigns colours by interpolating between two colours across the range of number of overlaps.'''
    def __init__(self, c1, c2, max_overlap):
        self.c1, self.c2 = c1, c2
        self.max_overlap = max_overlap
        
    def get_colour(self, n):
        def interpolate(fst, snd): return fst + (snd - fst) * (float(n) / self.max_overlap)
        return tuple(interpolate(fst, snd) for fst, snd in zip(self.c1, self.c2))
        
class TripleInterpolationMap(ColourMap):
    def __init__(self, c1, c2, c3, max_overlap):
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.max_overlap = max_overlap
        
    def get_colour(self, n):
        def interpolate(fst, snd, trd): 
            if(n > self.max_overlap // 2): return snd + (trd - snd) * (2.0 * (n - (n / 2.0))  / self.max_overlap) 
            else: return fst + (snd - fst) * (2.0 * n / self.max_overlap)
        return tuple(interpolate(fst, snd, trd) for fst, snd, trd in zip(self.c1, self.c2, self.c3))

def main():
    matplotlib.use('TKAgg') #qt is broken on one of my systems - this is workaround

    boxes = [generate_boxes(i, 80, 80) for i in range(1, 8)]
    shifts = ((xshift, yshift) for yshift in range(200, -1, -100) for xshift in range(0, 301, 100))
    for ls, (xshift, yshift) in zip(boxes, shifts): 
        for b in ls: b.shift(xshift=xshift, yshift=yshift)
    #colours = FixedMap([ColourMap.RED, ColourMap.ORANGE, ColourMap.YELLOW, ColourMap.GREEN, ColourMap.LIGHT_BLUE, ColourMap.INDIGO, ColourMap.VIOLET])
    #colours = InterpolationMap(ColourMap.PURE_BLUE, ColourMap.PURE_RED, 7)
    colours = TripleInterpolationMap(ColourMap.PURE_BLUE, ColourMap.PURE_GREEN, ColourMap.PURE_RED, 7)
    print(boxes)
    
    print()
    
    lgrid = LocatorGrid(0, 1440, 100, 0, 1500, 100)
    for ls in boxes:
        for b in ls: lgrid.split_all_boxes(b)
    print(lgrid.all_splits)

    fig,ax = plt.subplots(1)
    for ls in boxes:
        for n, b in enumerate(ls):
            ax.add_patch(patches.Rectangle((b.pt1.x, b.pt1.y), b.pt2.x - b.pt1.x, b.pt2.y - b.pt1.y, linewidth=1, ec="black", fc=colours.get_colour(n))) 
    ax.set_xlim([0, max(b.pt2.x for b in itertools.chain(*boxes))])
    ax.set_ylim([0, max(b.pt2.y for b in itertools.chain(*boxes))])
    plt.show()
    
    fig,ax = plt.subplots(1)
    for n, ls in enumerate(lgrid.all_splits):
        for b in ls: ax.add_patch(patches.Rectangle((b.pt1.x, b.pt1.y), b.pt2.x - b.pt1.x, b.pt2.y - b.pt1.y, linewidth=1, ec="black", fc=colours.get_colour(n)))
    ax.set_xlim([0, max(b.pt2.x for b in itertools.chain(*lgrid.all_splits))])
    ax.set_ylim([0, max(b.pt2.y for b in itertools.chain(*lgrid.all_splits))])
    plt.show()
    
    print()
    
    boxes = [GenericBox(0, 10, 0, 30), GenericBox(5, 15, 0, 30), GenericBox(0, 10, 15, 45), GenericBox(0, 17, 0, 30)]
    lgrid = LocatorGrid(0, 1440, 100, 0, 1500, 100)
    for b in boxes: lgrid.split_all_boxes(b)
    print(lgrid.all_splits)
    
    colours = FixedMap([ColourMap.RED, ColourMap.ORANGE, ColourMap.YELLOW, ColourMap.GREEN, ColourMap.LIGHT_BLUE, ColourMap.INDIGO, ColourMap.VIOLET])
    
    fig,ax = plt.subplots(1)
    for n, b in enumerate(list(reversed(boxes))):
        ax.add_patch(patches.Rectangle((b.pt1.x, b.pt1.y), b.pt2.x - b.pt1.x, b.pt2.y - b.pt1.y, linewidth=1, ec="black", fc=colours.get_colour(n))) 
    ax.set_xlim([0, max(b.pt2.x for b in boxes)])
    ax.set_ylim([0, max(b.pt2.y for b in boxes)])
    plt.show()
    
    fig,ax = plt.subplots(1)
    for n, ls in enumerate(lgrid.all_splits):
        for b in ls: ax.add_patch(patches.Rectangle((b.pt1.x, b.pt1.y), b.pt2.x - b.pt1.x, b.pt2.y - b.pt1.y, linewidth=1, ec="black", fc=colours.get_colour(n)))
    ax.set_xlim([0, max(b.pt2.x for b in itertools.chain(*lgrid.all_splits))])
    ax.set_ylim([0, max(b.pt2.y for b in itertools.chain(*lgrid.all_splits))])
    plt.show()
    
    print([colours.get_colour(i) for i in range(1, 8)])
    
if __name__ == "__main__": main()