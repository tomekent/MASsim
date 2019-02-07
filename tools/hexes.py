# Generated code -- CC0 -- No Rights Reserved -- http://www.redblobgames.com/grids/hexagons/

import collections
import math



# Point = collections.namedtuple("Point", ["x", "y"])

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class hexOrientation():
    def __init__(self, type):
        if type == "flat":
            self.initTypeFlat()
        else:
            self.initTypePointy()

    def initTypePointy(self):
        self.f0 = math.sqrt(3.0)
        self.f1 = math.sqrt(3.0)
        self.f2 = 0.0
        self.f3 = 3.0 / 2.0
        self.b0 = math.sqrt(3.0) / 3.0
        self.b1 = -1.0 / 3.0
        self.b2 = 0.0
        self.b3 = 2.0 / 3.0
        self.start_angle = 0.5

    def initTypeFlat(self):
        self.f0 = 3.0 / 2.0
        self.f1 = 0.0
        self.f2 = math.sqrt(3.0) / 2.0
        self.f3 = math.sqrt(3.0)
        self.b0 = 2.0 / 3.0
        self.b1 = 0.0
        self.b2 = -1.0 / 3.0
        self.b3 = math.sqrt(3.0) / 3.0
        self.start_angle = 0.0


class hex():
    def __init__(self, q, r, s=None):
        self.q = q
        self.r = r
        if s == None:
            self.s = - (q + r)
        else:
            self.s = s
        self.col = r
        self.row = q - (-math.floor(r/2))
        assert not (round(q + r + s) != 0), "q + r + s must be 0"

    def __add__(self, other):
        return hex(self.q + other.q, self.r + other.r, self.s + other.s)

    def __sub__(self, other):
        return hex(self.q - other.q, self.r - other.r, self.s - other.s)

    def __mul__(self, k):
        return hex(self.q * k, self.r * k, self.s * k)

    def rotate_left(self):
        return hex(-self.s, -self.q, -self.r)

    def rotate_right(self):
        return hex(-self.r, -self.s, -self.q)

    def neighbour(self, direction):
        if direction == 0:
            return self + hex(1, 0, -1)
        elif direction == 1:
            return self + hex(1, -1, 0)
        elif direction == 2:
            return self + hex(0, -1, 1)
        elif direction == 3:
            return self + hex(-1, 0, 1)
        elif direction == 4:
            return self + hex(-1, 1, 0)
        else:
            return self + hex(0, 1, -1)

    def __len__(self):
        return int((abs(self.q) + abs(self.r) + abs(self.s) ) /2)


def hex_distance(a, b):
    return len(a - b)


def hex_round(h):
    qi = int(round(h.q))
    ri = int(round(h.r))
    si = int(round(h.s))
    q_diff = abs(qi - h.q)
    r_diff = abs(ri - h.r)
    s_diff = abs(si - h.s)
    if q_diff > r_diff and q_diff > s_diff:
        qi = -ri - si
    elif r_diff > s_diff:
        ri = -qi - si
    else:
        si = -qi - ri
    return hex(qi, ri, si)

def hex_lerp(a, b, t):
    return hex(a.q * (1.0 - t) + b.q * t, a.r * (1.0 - t) + b.r * t, a.s * (1.0 - t) + b.s * t)

def hex_linedraw(a, b):
    N = hex_distance(a, b)
    a_nudge = hex(a.q + 0.000001, a.r + 0.000001, a.s - 0.000002)
    b_nudge = hex(b.q + 0.000001, b.r + 0.000001, b.s - 0.000002)
    results = []
    step = 1.0 / max(N, 1)
    for i in range(0, N + 1):
        results.append(hex_round(hex_lerp(a_nudge, b_nudge, step * i)))
    return results




# OffsetCoord = collections.namedtuple("OffsetCoord", ["col", "row"])
#
# EVEN = 1
# ODD = -1
# def qoffset_from_cube(offset, h):
#     col = h.q
#     row = h.r + (h.q + offset * (h.q & 1)) // 2
#     return OffsetCoord(col, row)
#
# def qoffset_to_cube(offset, h):
#     q = h.col
#     r = h.row - (h.col + offset * (h.col & 1)) // 2
#     s = -q - r
#     return Hex(q, r, s)
#
# def roffset_from_cube(offset, h):
#     col = h.q + (h.r + offset * (h.r & 1)) // 2
#     row = h.r
#     return OffsetCoord(col, row)
#
# def roffset_to_cube(offset, h):
#     q = h.col - (h.row + offset * (h.row & 1)) // 2
#     r = h.row
#     s = -q - r
#     return Hex(q, r, s)



class hexGrid:
    def __init__(self, origin, hexSize, gridSize):
        self.layout = hexLayout("flat", origin, hexSize)
        self.size = gridSize
        self.hexes = {} #dictionary to use hashes for keys
        self.hexinfo = {}
        self.generate_hexes()
        self.type = 'rectangle'
        self.get_mat_size()

    def generate_hexes(self):
        width = self.layout.size.x * 6 / 4.
        height = self.layout.size.y * math.sqrt(3)
        noY = math.ceil(self.size.x / width) + 1
        noX = math.ceil(self.size.y / height) + 1
        x0 = 0
        y0 = 0
        count = 0
        for x in range(x0, x0 + noX):
            for y in range(y0, y0 + noY):
                hi = self.qoffset_to_cube(x, y)
                newHex = hex(hi[0], hi[1], hi[2])
                # print(x,y,hi)
                self.set_hex(newHex)
                count += 1

    def get_hex(self, q, r, s):
        return self.hexes[hash((q,r, s))]

    def set_hex(self, newHex):
        self.hexes[(newHex.q, newHex.r)] = newHex
        self.hexinfo[(newHex.q, newHex.r)] = {'AgentCount': 0, 'TaskCount': 0, 'occupied': False, 'last_visited': 0.0, 'score': 0.0}

    def qoffset_from_cube(self, q, r, s):
        offset = -1 # even
        col = q
        row = r + (q + offset * (q & 1)) // 2
        return (col, row)

    def qoffset_to_cube(self, row, col):
        offset = -1  # even
        q = col
        r = row - (col + offset * (col & 1)) // 2
        s = -q - r
        return (q, r, s)

    def get_mat_size(self):
        N = len(self.hexes.items())
        rows = []
        cols = []
        for hexkey, hexi in self.hexes.items():
            rows.append(hexi.row)
            cols.append(hexi.col)

        self.row_off = -min(rows)
        self.col_off = -min(cols)
        self.n_max = max(rows) + self.row_off + 1
        self.m_max = max(cols) + self.col_off + 1



class hexLayout:
    def __init__(self, type, origin, size):
        self.orientation = hexOrientation(type)
        self.size = size # size of hexagon
        self.origin = origin
        self.cube_directions = [
            hex(+1, -1, 0), hex(+1, 0, -1), hex(0, +1, -1),
            hex(-1, +1, 0), hex(-1, 0, +1), hex(0, -1, +1),
        ]

    def hex_to_pixel(self, h):
        M = self.orientation
        size = self.size
        origin = self.origin
        x = (M.f0 * h.q + M.f1 * h.r) * size.x
        y = (M.f2 * h.q + M.f3 * h.r) * size.y
        return Point(x + origin.x, y + origin.y)

    def pixel_to_hex(self,  p):
        M = self.orientation
        size = self.size
        origin = self.origin
        pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
        q = M.b0 * pt.x + M.b1 * pt.y
        r = M.b2 * pt.x + M.b3 * pt.y
        # q = (2. / 3 * p.x) / size.x
        # r = (-1. / 3 * p.x + math.sqrt(3) / 3 * p.y) / size.y
        s = -(q + r)
        return hex_round(hex(q,r,s))
        # return hex(q, r, -q - r)

    def hex_corner_offset(self, corner):
        M = self.orientation
        size = self.size
        angle = 2.0 * math.pi * (M.start_angle - corner) / 6.0
        return Point(size.x * math.cos(angle), size.y * math.sin(angle))

    def polygon_corners(self, h):
        corners = []
        center = self.hex_to_pixel(h)
        for i in range(0, 6):
            offset = self.hex_corner_offset(i)
            corners.append(Point(center.x + offset.x, center.y + offset.y))
        return corners

    def hex_adjacent(self, h):
        adj = []
        for i in range(0, 6):
            adj.append(h + self.cube_directions[i])
        return adj




def main():

    import matplotlib.pyplot as plt


    a = hex(-1,-2,3)
    b = hex(1,-1,0)
    c = hex(-1,-1,2)
    d = hex(10, 5, -15)
    hexes = [a, b, c]
    hexes = [a]
    # for i in range(0, 6):
    #     hexes.append(a.neighbour(i))
    line = hex_linedraw(a, d)
    hexes.extend(line)
    # neb = c.neighbour(2)
    # for i in range(0, 4):
    #     neb = neb.neighbour(2)
    #     hexes.append(neb)
    # layout = hexLayout("flat", Point(0,0), Point(10, 10))
    ax = plt.subplot(111)
    # hexes.append(layout.pixel_to_hex(Point(75.5, 13.0)))
    # for hexi in hexes:
    #     corners = layout.polygon_corners(hexi)
    #     xs = [corner.x for corner in corners]
    #     ys = [corner.y for corner in corners]
    #     points = [[corner.x, corner.y] for corner in corners]
    #     print(points)
    #     hexpatch = plt.Polygon(points, fill=True)
    #     ax.add_patch(hexpatch)

    grid = hexGrid(Point(0.0 - 5.0, 0.0 - 5.0), Point(10, 10),  Point(100. + 0, 100.0 + 0.0))
    # grid = hexGrid(Point(-20,-20), Point(10, 10), Point(200, 200))

    hexes = []
    for hexkey, hexi in grid.hexes.items():
        corners = grid.layout.polygon_corners(hexi)
        points = [[corner.x, corner.y] for corner in corners]
        hexpatch = plt.Polygon(points, fill=True, alpha=0.25)
        ax.add_patch(hexpatch)
        center = grid.layout.hex_to_pixel(hexi)
        ax.annotate('%i, %i' %(hexi.q, hexi.r), (center.x, center.y), color='w', weight='bold',
                    fontsize=6, ha='center', va='center')

    hex1  = grid.layout.pixel_to_hex(Point(20, 20.0))
    hex2  = grid.layout.pixel_to_hex(Point(20, 80.0))
    hex3  = grid.layout.pixel_to_hex(Point(80, 20.0))
    hex4  = grid.layout.pixel_to_hex(Point(80, 80.0))
    hex5  = grid.layout.pixel_to_hex(Point(50, 50.0))
    hexkeys = [hexkey for hexkey in grid.hexes.keys()]

    adjs = grid.layout.hex_adjacent(hex1)
    for hexi in adjs:#[hex1, hex2, hex3, hex4, hex5]:
        corners = grid.layout.polygon_corners(hexi)
        points = [[corner.x, corner.y] for corner in corners]
        hexpatch = plt.Polygon(points, fill=True, alpha=0.95, color='red')
        ax.add_patch(hexpatch)
        # print(hexkeys.index((hexi.q, hexi.r)))
        # print(hexi.q, hexi.r)
    corners = grid.layout.polygon_corners(hex1)
    points = [[corner.x, corner.y] for corner in corners]
    hexpatch = plt.Polygon(points, fill=True, alpha=0.95, color='blue')
    ax.add_patch(hexpatch)

    hexi = grid.layout.pixel_to_hex(Point(00., 0.0))
    print(hexi.q, hexi.r)
    corners = grid.layout.polygon_corners(hexi)
    points = [[corner.x, corner.y] for corner in corners]
    hexpatch = plt.Polygon(points, fill=True, alpha=0.95, color='red')
    ax.add_patch(hexpatch)
    print(len(grid.hexes.items()))

    N = len(grid.hexes.items())
    rows = []
    cols = []
    for hexkey, hexi in grid.hexes.items():
        rows.append(hexi.row)
        cols.append(hexi.col)

    row_off = -min(rows)
    col_off = -min(cols)
    print(row_off,col_off)
    n_max = max(rows) + row_off + 1
    m_max = max(cols) + col_off + 1

    import numpy as np
    mat_col = np.zeros((n_max, m_max), dtype=np.int32)
    mat_row = np.zeros((n_max, m_max), dtype=np.int32)
    print(n_max,m_max)
    for i in range(N):
        mat_col[rows[i] + row_off][cols[i] + col_off] = 1#cols[i]
        mat_row[rows[i] + row_off][cols[i] + col_off] = 1#rows[i]

    print(mat_col)
    print(mat_row)

    ax.autoscale_view()
    plt.show()


if __name__ == "__main__":
    main()

