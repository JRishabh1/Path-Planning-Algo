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

# Node List
node_list = []

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

# Generate a middle point between the start/end points
def middle_point(image, x1, y1, x2, y2):
    average_x = (x1 + x2) / 2
    average_y = (y1 + y2) / 2
    variance = 1

    check = True
    middle_x = -1
    middle_y = -1

    while check:
        middle_x = int(round(random.normalvariate(average_x, variance)))
        middle_y = int(round(random.normalvariate(average_y, variance)))

        if(image[middle_y][middle_x][0] == 255):
            check = False
        
        variance = variance * 1.1
    
    return (middle_x, middle_y)

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
    x_coords = []
    if x1 == x2:
        x_coords = []
    else:
        x_coords = list(np.arange(x1, x2, (x2 - x1)/100))

    for x in x_coords:
        x_floor = math.floor(x)
        y_floor = math.floor( ((y2-y1)/(x2-x1))*(x-x1) + y1 )

        if(image[y_floor][x_floor][0] == 0):
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
    if y < 0 or y > height or x < 0 or x > length:
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(image, x, y, end[0], end[1]):
            directCon = False
        else:
            directCon = True
        
        # check connection between two nodes
        if collision(image, x, y, x2, y2):
            nodeCon = False
        else:
            nodeCon = True

    return (x, y, directCon, nodeCon)
    

# RRT Algorithm
def RRT(image, start, end, iterations, step_size, node_list):
    height = len(image)
    length = len(image[0])

    node_list.append(0)
    node_list[0] = Node(start[0], start[1])
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])

    total_iter = 0
    i = 1
    pathFound = False
    while pathFound == False:
        total_iter = total_iter + 1

        if(total_iter == iterations):
            print("Iteration limit exceeded.")
            break

        # Get random point
        new_x, new_y = random_point(image, height, length)

        # Find the nearest node
        nearest_ind = nearest_node(new_x, new_y, node_list)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y

        # Check connection(s)
        tx, ty, directCon, nodeCon = check_collision(image, new_x, new_y, nearest_x, nearest_y, end, step_size)

        if directCon and nodeCon:
            node_list.append(i)
            node_list[i] = Node(tx, ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)

            # Just create straight line, major performance improvement
            pathFound = True
            print("Path has been found after " + str(total_iter) + " iterations!")
            
            return node_list[i].parent_x, node_list[i].parent_y, i
            
        elif nodeCon:
            node_list.append(i)
            node_list[i] = Node(tx, ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            i = i + 1
            continue
        else:     
            continue

def draw_result(image, result, start, end, parent_x_array, parent_y_array, step_size, index, node_list):
    # Get height and length of image
    height = len(image)
    length = len(image[0])

    # Draw all of the nodes in the RRT
    for node in node_list:
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
    for j in range(len(parent_x_array)):
        parent_x = math.floor(parent_x_array[j])
        parent_y = math.floor(parent_y_array[j])

        for x in range(parent_x - 2, parent_x + 2):
            for y in range(parent_y - 2, parent_y + 2):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (127, 127, 127))
    
    # Draw straight line between last node and end point
    last_node = node_list[index]
    draw_x = last_node.x
    draw_y = last_node.y
    _, theta = dist_and_angle(draw_x, draw_y, end[0], end[1])
    drawing_check = True
    while drawing_check:
        draw_x = draw_x + step_size * np.cos(theta)
        draw_y = draw_y + step_size * np.sin(theta)

        line_x = math.floor(draw_x)
        line_y = math.floor(draw_y)
        for x in range(line_x - 2, line_x + 2):
            for y in range(line_y - 2, line_y + 2):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (127, 127, 127))

        if end[0]-5 < draw_x and draw_x<end[0]+5 and end[1]-5 < draw_y and draw_y < end[1]+5:
            drawing_check = False


def main():
    node_list = []

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

    middle = middle_point(image, start_first_number, start_second_number, end_first_number, end_second_number)

    iterations = int(input('Enter the number of iterations (10000 recommended): '))
    step_size = int(input('Enter the step size of the RRT (3 recommended): '))

    result = im.copy() # result image
    
    parent_x_array1, parent_y_array1, index1 = RRT(image, start, middle, iterations, step_size, node_list)
    draw_result(image, result, start, middle, parent_x_array1, parent_y_array1, step_size, index1, node_list)
    first_node_list = node_list.copy()
    node_list = []

    parent_x_array2, parent_y_array2, index2 = RRT(image, end, middle, iterations, step_size, node_list)
    draw_result(image, result, end, middle, parent_x_array2, parent_y_array2, step_size, index2, node_list)
    second_node_list = node_list.copy()
    node_list = []
    
    result.show()

if __name__ == '__main__':
    main()
