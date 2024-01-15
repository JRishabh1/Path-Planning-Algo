# Implement RRT and RRT*

import numpy as np 
import math 
import random
from PIL import Image

# Get image and convert to 2D array
im = Image.open("world3.png")
result = im.copy() # result image

image = np.asarray(im)

start = (5, 5)
end = (670, 370)

class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Node List
node_list = []

# Generate a random point in the maze
def random_point(height, length):
    check = True
    new_x = -1
    new_y = -1

    while check:
        new_x = random.randint(0, length - 1)
        new_y = random.randint(0, height - 1)

        if(image[new_y][new_x][0] == 255):
            check = False
    
    return (new_x, new_y)

# Return the distance and angle between the new point and nearest node
def dist_and_angle(x1, y1, x2, y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2 - y1, x2 - x1)
    return (dist, angle)

# Return the nearest node's index
def nearest_node(x, y):
    temp_dist = []
    for i in range(len(node_list)):
        dist, _ = dist_and_angle(x, y, node_list[i].x, node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))


# Check for collision(s)
# maybe make x2 > x1, y2 > y1
def collision(x1, y1, x2, y2):
    x_coords = list(np.arange(x1, x2, (x2 - x1)/100))

    for x in x_coords:
        x_floor = math.floor(x)
        y_floor = math.floor( ((y2-y1)/(x2-x1))*(x-x1) + y1 )

        if(image[y_floor][x_floor][0] == 0):
            return True # collision
        
    return False # no collision
    
def check_collision(x1, y1, x2, y2):
    _, theta = dist_and_angle(x2, y2, x1, y1)
    x = x2 + np.cos(theta)
    y = y2 + np.sin(theta)

    height = len(image)
    length = len(image[0])

    if y < 0 or y > height or x < 0 or x > length:
        # print("Point out of image bound")
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(x, y, end[0], end[1]):
            directCon = False
        else:
            directCon = True
        
        # check connection between two nodes
        if collision(x, y, x2, y2):
            nodeCon = False
        else:
            nodeCon = True

    return (x, y, directCon, nodeCon)
    

# RRT Algorithm
def RRT(image, start, end):
    height = len(image)
    length = len(image[0])
    # print(height, length)

    node_list.append(0)
    node_list[0] = Node(start[0], start[1])

    i = 1
    pathFound = False
    while pathFound == False:
        new_x, new_y = random_point(height, length)
        # print("Random points:", new_x, new_y)

        # Find the nearest node
        nearest_ind = nearest_node(new_x, new_y)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y
        # print("Nearest node coordinates:", nearest_x, nearest_y)


        # Check connection(s)
        tx, ty, directCon, nodeCon = check_collision(new_x, new_y, nearest_x, nearest_y)

        if directCon and nodeCon:
            # print("Node can connect directly with end")
            node_list.append(i)
            node_list[i] = Node(tx, ty)

            # print("Path has been found")
            pathFound = True

            # Create line between nearest_node and endpoint
            for x in range(math.floor(nearest_x), end[0]):
                y_floor = math.floor( ((end[1]-nearest_y)/(end[0]-nearest_x))*(x-nearest_x) + nearest_y )
                node_list.append(Node(x, y_floor))

            # print('c')

            break

        elif nodeCon:
            # print("Nodes connected")
            node_list.append(i)
            node_list[i] = Node(tx, ty)
            i = i + 1

            # print('b')
            
            continue

        else:
            # print("No direct connection and no node connection.")
            # print("Generating new random numbers.")
            
            # print('a')
            
            continue

def main():
    RRT(image, start, end)

    for node in node_list:
        round_x = round(node.x)
        round_y = round(node.y)
        for x in range(round_x, round_x + 1):
            for y in range(round_y, round_y + 1):
                result.putpixel((x, y), (255, 0, 0))

    for x in range(start[0], start[0] + 1):
        for y in range(start[1], start[1] + 1):
            result.putpixel((x, y), (0, 0, 255))

    for x in range(end[0], end[0] + 1):
        for y in range(end[1], end[1] + 1):
            result.putpixel((x, y), (0, 0, 255))
        
    result.show()

if __name__ == '__main__':
    main()
