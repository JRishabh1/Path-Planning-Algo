# Implement RRT and RRT*

# Installed VSCode Code Runner extension and checked "Run in Terminal"

import numpy as np # NumPy
import math # Math
import random # Random
from PIL import Image # Image
import cv2 # CV2

class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = []

# Generate a random point in the maze
def random_point(image, height, length):
    check = True
    new_x = -1
    new_y = -1

    while check:
        new_x = random.randint(0, length - 1)
        new_y = random.randint(0, height - 1)

        if(image[new_y][new_x][0] > 220):
            check = False
    
    return (new_x, new_y)

# Return the distance and angle between the new point and nearest node
def dist_and_angle(x1, y1, x2, y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2 - y1, x2 - x1)
    return (dist, angle)

# Return the nearest node's index
def nearest_node(node_list, x, y):
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
        x_coords = list(np.arange(x1, x2, (x2 - x1)/100))

    for x in x_coords:
        x_round = round(x)
        y_round = round( ((y2-y1)/(x2-x1))*(x-x1) + y1 )

        if(y_round < 0 or y_round > height-1 or x_round < 0 or x_round > length-1 or image[y_round][x_round][0] == 0):
            return True # collision
        
    return False # no collision
    
def check_collision(image, x1, y1, x2, y2, end, step_size):
    height = len(image)
    length = len(image[0])

    # Point out of image bound
    if y1 < 0 or y2 > height-1 or x1 < 0 or x2 > length-1:
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(image, x1, y1, end[0], end[1]):
            directCon = False
        else:
            directCon = True
        
        # check connection between two nodes
        if collision(image, x1, y1, x2, y2):
            nodeCon = False
        else:
            nodeCon = True

    return (directCon, nodeCon)

# RRT Algorithm
def RRT(image, node_list, potential_map, boundary, start, end, iterations, step_size):
    # Height and length for image
    height = len(image)
    length = len(image[0])

    # Kernel for Laplace equation
    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])

    node_list.append(0)
    node_list[0] = Node(start[0], start[1])
    node_list[0].parent.append(node_list[0])

    total_iter = 0
    i = 1
    while True:
        total_iter = total_iter + 1

        if(total_iter >= iterations):
            print("Iteration limit exceeded.")
            return []
            # break

        # Get random point
        new_x, new_y = random_point(image, height, length)

        # Find the nearest node
        nearest_ind = nearest_node(node_list, new_x, new_y)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y

        # Check connection(s)
        _, nodeCon = check_collision(image, new_x, new_y, nearest_x, nearest_y, end, step_size)

        if nodeCon:
            # Laplace Equation
            map = potential_map.copy()
        
            for j in range(250): 
                map = cv2.filter2D(map, -1, kernel)

                # Reset boundary points
                map[new_y][new_x] = 0 # Goal
                for b in range(len(boundary)):
                    map[boundary[b].y][boundary[b].x] = 1
                for y in range(height):
                    map[y][0] = 1
                    map[y][length-1] = 1
                for x in range(length):
                    map[0][x] = 1
                    map[height-1][x] = 1

            # Gradient Descent
            poser = nearest_y
            posec = nearest_x

            limit = 0
            while True:
                if limit > 1000:
                    # print('Path not found! Maybe try providing more Laplace iterations!')
                    break

                gradr = map[round(poser+1)][round(posec)] - map[round(poser-1)][round(posec)]
                gradc = map[round(poser)][round(posec+1)] - map[round(poser)][round(posec-1)]
                maggrad = math.sqrt(gradr**2 + gradc**2)

                if(maggrad != 0):
                    a = step_size/maggrad # scale to pixel
                    poser = poser-a*gradr
                    posec = posec-a*gradc
                
                node_list.append(i)
                node_list[i] = Node(posec, poser)
                i = i + 1

                if(end[1]-step_size-1 < poser and poser < end[1]+step_size+1 and end[0]-step_size-1 < posec and posec < end[0]+step_size+1):
                    print("Number of iterations: " + str(total_iter))
                    return []

                if(new_y-step_size-1 < poser and poser < new_y+step_size+1 and new_x-step_size-1 < posec and posec < new_x+step_size+1):
                    break

                limit = limit + 1
            

def draw_result(image, node_list, result, start, end, parent_array):
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

    # Draw the path from the start point to the end point (IMPLEMENT THIS LATER!!)
    # for j in range(len(parent_array)):
    #     parent_x = math.floor(parent_array[j].x)
    #     parent_y = math.floor(parent_array[j].y)

    #     for x in range(parent_x - 2, parent_x + 2):
    #         for y in range(parent_y - 2, parent_y + 2):
    #             if(0 < x and x < length - 1 and 0 < y and y < height - 1):
    #                 result.putpixel((x, y), (127, 127, 127))

def main():
    conda = input("This is just for VSCode using the Conda Python version, type anything here: ")
    image = input("Enter the name of your image file (include .jpg, .png, etc.): ")

    # Get image and convert to 2D array
    im = Image.open(image)

    im = im.resize((169, 91)) # just to make the image smaller

    image = np.asarray(im)

    height = len(image)
    length = len(image[0])

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

    iterations = int(input('Enter the number of RRT iterations (10000 recommended): '))
    step_size = int(input('Enter the step size of the RRT (3 recommended): '))

    node_list = [] # Node List!

    potential_map = np.ones((height, length)) # add 2 to both dimensions?

    # Get the black walls
    boundary = []
    for y in range(height):
        for x in range(length):
            if(image[y][x][0] == 0):
                boundary.append(Node(x, y))

    parent_array = RRT(image, node_list, potential_map, boundary, start, end, iterations, step_size)

    result = im.copy() # result image

    draw_result(image, node_list, result, start, end, parent_array)


    result = result.resize((650, 350)) # make image large again

    result.show()


if __name__ == '__main__':
    main()
