import math

def distance(point1, point2):
    # Euclidean distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def is_closer(leftPoint, rightPoint, testPoint):
    distanceR = distance(rightPoint, testPoint)
    distanceL = distance(leftPoint, testPoint)

    if distanceL > distanceR:
        return 'Left'
    elif distanceL < distanceR:
        return 'Right'
    else:
        return 'None'

'''
# Example points
A = (1, 1)
B = (4, 5)
C = (7, 3)

# Check if A is closer to B than it is to C
if is_closer(A, B, C):
    print("A is closer to B than it is to C")
else:
    print("A is closer to C than it is to B")
'''