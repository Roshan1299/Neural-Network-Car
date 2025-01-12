import pygame
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def lerp(A, B, t):
    return A + (B - A) * t


def get_intersection(A, B, C, D):
    Ax, Ay = (A.x, A.y) if hasattr(A, 'x') else A
    Bx, By = (B.x, B.y) if hasattr(B, 'x') else B
    Cx, Cy = (C.x, C.y) if hasattr(C, 'x') else C
    Dx, Dy = (D.x, D.y) if hasattr(D, 'x') else D

    t_top = (Dx - Cx) * (Ay - Cy) - (Dy - Cy) * (Ax - Cx)
    u_top = (Cy - Ay) * (Ax - Bx) - (Cx - Ax) * (Ay - By)
    bottom = (Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay)

    if bottom != 0:
        t = t_top / bottom
        u = u_top / bottom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return {'point': Point(lerp(Ax, Bx, t), lerp(Ay, By, t)), 'offset': t}

    return None


def polys_intersect(poly_1, poly_2):
    for i in range(len(poly_1)):
        for j in range(len(poly_2)):
            touch = get_intersection(
                poly_1[i],
                poly_1[(i+1) % len(poly_1)],
                poly_2[j],
                poly_2[(i+j) % len(poly_2)]
            )
            if touch:
                return True
    return False


def get_rgba(value):
    normalized_value = (value + 1) / 2
    alpha = int(normalized_value * 255)

    if value >= 0:
        r = int(normalized_value * 255)
        g = int(normalized_value * 250)
        b = 55
    else:
        r = 0
        g = 50
        b = int((1 - normalized_value) * 255)

    return (r, g, b, alpha)


def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=10, offset=0):
    x1, y1 = start_pos
    x2, y2 = end_pos
    dl = dash_length

    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dash_count = int(length / dl)
    for i in range(dash_count):
        start = (x1 + (x2 - x1) * (i + offset) / dash_count,
                 y1 + (y2 - y1) * (i + offset) / dash_count)
        end = (x1 + (x2 - x1) * (i + 0.5 + offset) / dash_count,
               y1 + (y2 - y1) * (i + 0.5 + offset) / dash_count)
        pygame.draw.line(surface, color, start, end, width)
