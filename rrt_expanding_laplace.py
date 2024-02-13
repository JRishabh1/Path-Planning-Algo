# Implement RRT and RRT*

# Installed VSCode Code Runner extension and checked "Run in Terminal"

import numpy as np 
import math 
import random
from PIL import Image
import cv2

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
def random_point(start, height, length, potential_map):
    new_x = random.randint(0, length - 1)#start[1]
    new_y = random.randint(0, height - 1)#start[0]
    if(random.random() > 0.95):
        new_x = start[1]
        new_y = start[0]

    # needToFindNew = False
    # for node in node_list:
    #         if(int(node.y) == int(new_y) and int(node.x) == int(new_x)):
    #             needToFindNew = True

    while potential_map[new_y][new_x] == 1: #or needToFindNew
        new_x = random.randint(0, length - 1)
        new_y = random.randint(0, height - 1)
        # needToFindNew = False
        # for node in node_list:
        #     if(int(node.y) == int(new_y) and int(node.x) == int(new_x)):
        #         needToFindNew = True
    
    
    return (new_x, new_y, potential_map)

def LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list):
    potential_map = cv2.filter2D(potential_map, -1, kernel)

    # Reset boundary points
    potential_map[end[1]][end[0]] = 0 # Goal
    for node in node_list:
        potential_map[min(int(node.y), height - 1)][min(int(node.x), length - 1)] = 0
    for b in range(len(boundary_y)):
        potential_map[boundary_y[b]][boundary_x[b]] = 1
    for y in range(height):
        potential_map[y][0] = 1
        potential_map[y][length-1] = 1
    for x in range(length):
        potential_map[0][x] = 1
        potential_map[height-1][x] = 1
    return potential_map

# RRT Algorithm
def RRT(image, start, end, iterations, step_size):
    height = len(image)
    length = len(image[0])

    node_list.append(Node(end[0], end[1]))
    node_list[0].parent_x.append(end[0])
    node_list[0].parent_y.append(end[1])

    total_iter = 0
    i = 1
    pathFound = False

    potential_map = np.ones((height, length))
    potential_map[end[1]][end[0]] = 0 # Goal
    # Get the black walls
    boundary_x = []
    boundary_y = []
    for y in range(height):
        for x in range(length):
            if(image[y][x][0] == 0):
                boundary_x.append(x)
                boundary_y.append(y)

    # Kernel for Laplace equation
    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])
    for _ in range(5000):
        potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list)
    while pathFound == False:
        total_iter = total_iter + 1

        if(total_iter == iterations):
            print("Iteration limit exceeded.")
            return node_list#node_list[i - 1].parent_x, node_list[i - 1].parent_y
            #break

        # Get random point
        new_x, new_y, potential_map = random_point(start, height, length, potential_map)
        
        for _ in range(250):
            potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list)

        new_node_list = try_grad_descent(potential_map, step_size, new_x, new_y, node_list)
        i = i + len(new_node_list)
        node_list.extend(new_node_list)
        if len(new_node_list) != 0 and int(new_x) == start[0] and int(new_y) == start[1]:
            pathFound = True
    return node_list

def try_grad_descent(potential_map, step_size, new_x, new_y, node_list):
    poser = new_y
    posec = new_x
    toAppend = Node(int(posec), int(poser))
        # toAppend.parent_x.append(node_list[limit].x)
        # toAppend.parent_y.append(node_list[limit].u)
    

    limit = 0
    new_node_list = []
    new_node_list.append(toAppend)
    while True:
        if limit > 1000:
            # print('Path not found! Maybe try providing more Laplace iterations!')
            print("Couldn't find - Stuck :(")
            break

        gradr = potential_map[round(poser+1)][round(posec)] - potential_map[round(poser-1)][round(posec)]
        gradc = potential_map[round(poser)][round(posec+1)] - potential_map[round(poser)][round(posec-1)]
        maggrad = math.sqrt(gradr**2 + gradc**2)

        if(maggrad != 0):
            a = step_size/maggrad # scale to pixel
            poser = poser-a*gradr
            posec = posec-a*gradc
        print(poser)
        print(posec)
        toAppend = Node(int(posec), int(poser))
        # toAppend.parent_x.append(node_list[limit].x)
        # toAppend.parent_y.append(node_list[limit].u)
        new_node_list.append(toAppend)
        limit = limit + 1

        # if(potential_map[int(poser)][int(posec)] == 0):
        #     print("Found from ")
        #     print(new_x)
        #     print(new_y)
        #     print("to")
        #     print(poser)
        #     print(posec)
        #     return new_node_list
            
        for node in node_list:
            if(node.y-step_size-1 < poser and poser < node.y+step_size+1 and node.x-step_size-1 < posec and posec < node.x+step_size+1):
                
                print("Found from ")
                print(new_x)
                print(new_y)
                print("to")
                print(node.x)
                print(node.y)
                return new_node_list
               # break

        # if(new_y-step_size-1 < poser and poser < new_y+step_size+1 and new_x-step_size-1 < posec and posec < new_x+step_size+1):
        #     print("Stuck :(")
        #     return []


# def draw_result(image, result, start, end, parent_x_array, parent_y_array):
def draw_result(image, result, start, end, node_list):
    height = len(image)
    length = len(image[0])

    # Draw all of the nodes in the RRT
    for node in node_list:
        print(node)
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
                
def main():
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

    node_list = RRT(image, start, end, iterations, step_size)#parent_x_array, parent_y_array = RRT(image, start, end, iterations, step_size)

    result = im.copy() # result image
    draw_result(image, result, start, end, node_list)#draw_result(image, result, start, end, parent_x_array, parent_y_array)
    result.show()

if __name__ == '__main__':
    main()
