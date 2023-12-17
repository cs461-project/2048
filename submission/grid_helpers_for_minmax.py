import math


#direction UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3


def getAvailableChildren(grid):
    # gets all children and the moving directions
    allmoves = [0, 1, 2, 3]
    children = []
    moving = []
    for m in allmoves:
        gridcopy = list(grid)
        moved = move(gridcopy, m)
        # move method returns True if moved and makes the change to gridcopy itself
        if moved == True:
            children.append(gridcopy)
            moving.append(m)
    return [children, moving]

def move(grid, direction):
    # if there is a move it is changed and return is True
    moved = False
    if direction == 0:
        # UP
        for i in range(4):
            cells = []
            # cells has all elements for a column from top to bottom
            for j in range(i, i + 13, 4):
                cell = grid[j]
                if cell != 0:
                    cells.append(cell)
            merge(cells)
            for j in range(i, i + 13, 4):
                value = cells.pop(0) if cells else 0
                if grid[j] != value:
                    moved = True
                grid[j] = value
        return moved
    elif direction == 1:
        # DOWN
        for i in range(4):
            cells = []
            # cells has all elements of column from bottom to top
            for j in range(i + 12, i - 1, -4):
                cell = grid[j]
                if cell != 0:
                    cells.append(cell)
            merge(cells)
            for j in range(i + 12, i - 1, -4):
                value = cells.pop(0) if cells else 0
                if grid[j] != value:
                    moved = True
                grid[j] = value
        return moved
    elif direction == 2:
        # LEFT
        for i in [0, 4, 8, 12]:
            cells = []
            # cells has all elements of a row from left to right
            for j in range(i, i + 4):
                cell = grid[j]
                if cell != 0:
                    cells.append(cell)
            merge(cells)
            for j in range(i, i + 4):
                value = cells.pop(0) if cells else 0
                if grid[j] != value:
                    moved = True
                grid[j] = value
        return moved
    elif direction == 3:
        # RIGHT
        for i in [3, 7, 11, 15]:
            cells = []
            # cells has all elements of a row from right to left
            for j in range(i, i - 4, -1):
                cell = grid[j]
                if cell != 0:
                    cells.append(cell)
            merge(cells)
            for j in range(i, i - 4, -1):
                value = cells.pop(0) if cells else 0
                if grid[j] != value:
                    moved = True
                grid[j] = value
        return moved

def canMove(grid):
    if 0 in grid:
        # if there is an empty space in the grid
        return True
    for i in range(16):
        if (i + 1) % 4 != 0:
            # for all elements except the last column
            # if any element has same right element
            if grid[i] == grid[i + 1]:
                return True
        if i < 12:
            # for all except last row elements
            # if any element has same below element
            if grid[i] == grid[i + 4]:
                return True
    return False



def merge(cells):
    # merges the cells and sends back in order to be inserted
    if len(cells) <= 1:
        return cells
    i = 0
    while i < len(cells) - 1:
        if cells[i] == cells[i + 1]:
            cells[i] *= 2
            del cells[i + 1]
        i += 1



