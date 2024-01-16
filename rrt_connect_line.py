# Implement RRT and RRT*

# Installed VSCode Code Runner extension and checked "Run in Terminal"
# I don't think this is the complete intended RRT Connect, will edit more later

import numpy as np 
import math 
import random
from PIL import Image

class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

# Generate a random point in the maze
def random_point(image, height, length):
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
def nearest_node(x, y, node_list):
    temp_dist = []
    for i in range(len(node_list)):
        dist, _ = dist_and_angle(x, y, node_list[i].x, node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))


# Check for collision(s)
def collision(image, x1, y1, x2, y2):
    height = len(image)
    length = len(image[0])

    x_coords = []
    if x1 == x2:
        x_coords = []
    else:
        x_coords = list(np.arange(x1, x2, (x2 - x1)/200))

    for x in x_coords:
        x_round = round(x)
        y_round = round( ((y2-y1)/(x2-x1))*(x-x1) + y1 )

        if(y_round < 0 or y_round > height-1 or x_round < 0 or x_round > length-1 or image[y_round][x_round][0] == 0):
            return True # collision
        
    return False # no collision
    
def check_collision(image, x1, y1, x2, y2, end, step_size):
    _, theta = dist_and_angle(x2, y2, x1, y1)
    # Step Size: 3
    x = x2 + step_size * np.cos(theta)
    y = y2 + step_size * np.sin(theta)

    height = len(image)
    length = len(image[0])

    # Point out of image bound
    if y < 0 or y > height-1 or x < 0 or x > length-1:
        directCon = False
        nodeCon = False
    else:
        # check connection between two nodes
        if collision(image, x, y, x2, y2):
            nodeCon = False
        else:
            nodeCon = True

    return (x, y, False, nodeCon)
    

# RRT Algorithm
def RRT_Connect(image, start, end, iterations, step_size, node_list1, node_list2):
    height = len(image)
    length = len(image[0])

    node_list1.append(0)
    node_list1[0] = Node(start[0], start[1])
    node_list1[0].parent_x.append(start[0])
    node_list1[0].parent_y.append(start[1])

    node_list2.append(0)
    node_list2[0] = Node(end[0], end[1])
    node_list2[0].parent_x.append(end[0])
    node_list2[0].parent_y.append(end[1])

    total_iter = 0
    i1 = 1
    i2 = 1
    pathFound = False
    while pathFound == False:
        total_iter = total_iter + 1

        if(total_iter >= iterations):
            print("Iteration limit reached.")
            break

        # Get random points
        new1_x, new1_y = random_point(image, height, length)
        new2_x, new2_y = random_point(image, height, length)

        # Find the nearest node
        nearest1_ind = nearest_node(new1_x, new1_y, node_list1)
        nearest1_x = node_list1[nearest1_ind].x
        nearest1_y = node_list1[nearest1_ind].y
        nearest2_ind = nearest_node(new2_x, new2_y, node_list2)
        nearest2_x = node_list2[nearest2_ind].x
        nearest2_y = node_list2[nearest2_ind].y

        # Check connection(s)
        tx1, ty1, _, nodeCon1 = check_collision(image, new1_x, new1_y, nearest1_x, nearest1_y, end, step_size)
        tx2, ty2, _, nodeCon2 = check_collision(image, new2_x, new2_y, nearest2_x, nearest2_y, start, step_size)

        # Maybe get nearest_node instead of just growing both trees like this
        connection = not collision(image, tx1, ty1, tx2, ty2)

        if nodeCon1 and nodeCon2 and connection:
            node_list1.append(i1)
            node_list1[i1] = Node(tx1, ty1)
            node_list1[i1].parent_x = node_list1[nearest1_ind].parent_x.copy()
            node_list1[i1].parent_y = node_list1[nearest1_ind].parent_y.copy()
            node_list1[i1].parent_x.append(tx1)
            node_list1[i1].parent_y.append(ty1)

            node_list2.append(i2)
            node_list2[i2] = Node(tx2, ty2)
            node_list2[i2].parent_x = node_list2[nearest2_ind].parent_x.copy()
            node_list2[i2].parent_y = node_list2[nearest2_ind].parent_y.copy()
            node_list2[i2].parent_x.append(tx2)
            node_list2[i2].parent_y.append(ty2)

            # Just create straight line, major performance improvement
            pathFound = True
            print("Path has been found after " + str(total_iter) + " iterations!")
            
            return node_list1[i1].parent_x, node_list1[i1].parent_y, i1, node_list2[i2].parent_x, node_list2[i2].parent_y, i2
        
        if nodeCon1:
            node_list1.append(i1)
            node_list1[i1] = Node(tx1, ty1)
            node_list1[i1].parent_x = node_list1[nearest1_ind].parent_x.copy()
            node_list1[i1].parent_y = node_list1[nearest1_ind].parent_y.copy()
            node_list1[i1].parent_x.append(tx1)
            node_list1[i1].parent_y.append(ty1)
            i1 = i1 + 1

        if nodeCon2:
            node_list2.append(i2)
            node_list2[i2] = Node(tx2, ty2)
            node_list2[i2].parent_x = node_list2[nearest2_ind].parent_x.copy()
            node_list2[i2].parent_y = node_list2[nearest2_ind].parent_y.copy()
            node_list2[i2].parent_x.append(tx2)
            node_list2[i2].parent_y.append(ty2)
            i2 = i2 + 1

def draw_result(image, result, start, end, parent_x_array1, parent_y_array1, parent_x_array2, parent_y_array2, step_size, index1, index2, node_list1, node_list2):
    # Get height and length of image
    height = len(image)
    length = len(image[0])

    # Draw all of the nodes in the RRT
    for node in node_list1:
        round_x = math.floor(node.x)
        round_y = math.floor(node.y)
        for x in range(round_x - 1, round_x + 1):
            for y in range(round_y - 1, round_y + 1):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (255, 0, 0))
    for node in node_list2:
        round_x = math.floor(node.x)
        round_y = math.floor(node.y)
        for x in range(round_x - 1, round_x + 1):
            for y in range(round_y - 1, round_y + 1):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (255, 0, 0))

    # Draw the start point
    for x in range(start[0] - 3, start[0] + 3):
        for y in range(start[1] - 3, start[1] + 3):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 255, 0))

    # Draw the end point
    for x in range(end[0] - 3, end[0] + 3):
        for y in range(end[1] - 3, end[1] + 3):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 0, 255))

    # Draw the path from the start point to the end point
    for j in range(len(parent_x_array1)):
        parent_x = math.floor(parent_x_array1[j])
        parent_y = math.floor(parent_y_array1[j])

        for x in range(parent_x - 2, parent_x + 2):
            for y in range(parent_y - 2, parent_y + 2):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (127, 127, 127))
    for k in range(len(parent_x_array2)):
        parent_x = math.floor(parent_x_array2[k])
        parent_y = math.floor(parent_y_array2[k])

        for x in range(parent_x - 2, parent_x + 2):
            for y in range(parent_y - 2, parent_y + 2):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (127, 127, 127))
    
    # Draw straight line between last nodes of the two trees
    last_node1 = node_list1[index1]
    last_node2 = node_list2[index2]
    draw1_x = last_node1.x
    draw1_y = last_node1.y
    draw2_x = last_node2.x
    draw2_y = last_node2.y
    _, theta = dist_and_angle(draw1_x, draw1_y, draw2_x, draw2_y)
    drawing_check = True
    while drawing_check:
        draw1_x = draw1_x + step_size * np.cos(theta)
        draw1_y = draw1_y + step_size * np.sin(theta)

        line_x = math.floor(draw1_x)
        line_y = math.floor(draw1_y)
        for x in range(line_x - 2, line_x + 2):
            for y in range(line_y - 2, line_y + 2):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (127, 127, 127))

        if draw2_x-3 < draw1_x and draw1_x<draw2_x+3 and draw2_y-3 < draw1_y and draw1_y < draw2_y+3:
            drawing_check = False


def main():
    node_list1 = []
    node_list2 = []

    conda = input("This is just for VSCode using the Conda Python version, type anything here: ")
    image = input("Enter the name of your image file (include .jpg, .png, etc.): ")

    # Get image and convert to 2D array
    im = Image.open(image)
    image = np.asarray(im)

    start = input('Enter your start point in the form of "(x, y)": ')
    end = input('Enter your end point in the form of "(x, y)": ')

    start_comma = start.index(',')
    end_comma = end.index(',')

    start_first_number = int(start[1:start_comma])
    start_second_number = int(start[start_comma+2:len(start)-1])
    end_first_number = int(end[1:end_comma])
    end_second_number = int(end[end_comma+2:len(end)-1])

    start = (start_first_number, start_second_number)
    end = (end_first_number, end_second_number)

    iterations = int(input('Enter the number of iterations (10000 recommended): '))
    step_size = int(input('Enter the step size of the RRT (3 recommended): '))

    result = im.copy() # result image
    
    parent_x_array1, parent_y_array1, index1, parent_x_array2, parent_y_array2, index2 = RRT_Connect(image, start, end, iterations, step_size, node_list1, node_list2)
    draw_result(image, result, start, end, parent_x_array1, parent_y_array1, parent_x_array2, parent_y_array2, step_size, index1, index2, node_list1, node_list2)
    
    result.show()

if __name__ == '__main__':
    main()
