# Implement RRT and RRT*

# Installed VSCode Code Runner extension and checked "Run in Terminal"

import numpy as np 
import math 
import random
from PIL import Image
import cv2
import imageio
import csv


# Len of each iteration
len_per_iter = []

class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

# Node List
node_list = []

# Results
result_images = []

# Generate a random point in the maze
def random_point(start, height, length, potential_map, file_name, image, end, prob_of_choosing_start, show_every_attempted_point, show_expansion):
    new_x = random.randint(0, length - 1)#start[1]
    new_y = random.randint(0, height - 1)#start[0]
    if(random.random() > 1 - prob_of_choosing_start):
        new_x = start[1]
        new_y = start[0]
    if(show_every_attempted_point and image[new_y][new_x][0] == 255):
        im = Image.open(file_name)
        result = im.copy() # result image
        draw_result(image, result, start, end, node_list, potential_map, show_expansion)#draw_result(image, result, start, end, parent_x_array, parent_y_array)
        for x in range(new_x - 3, new_x + 3):
            for y in range(new_y - 3, new_y + 3):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (0, 0, 255))
        result_images.append(result)
    # needToFindNew = False
    # for node in node_list:
    #         if(int(node.y) == int(new_y) and int(node.x) == int(new_x)):
    #             needToFindNew = True

    while potential_map[new_y][new_x] == 1: #or needToFindNew
        new_x = random.randint(0, length - 1)
        new_y = random.randint(0, height - 1)
        if(show_every_attempted_point and image[new_y][new_x][0] == 255):
            im = Image.open(file_name)
            result = im.copy() # result image
            draw_result(image, result, start, end, node_list, potential_map, show_expansion)#draw_result(image, result, start, end, parent_x_array, parent_y_array)
            for x in range(new_x - 3, new_x + 3):
                for y in range(new_y - 3, new_y + 3):
                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                        result.putpixel((x, y), (0, 0, 255))
            result_images.append(result)
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
def RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion):
    height = len(image)
    length = len(image[0])

    node_list.append(Node(end[0], end[1]))
    node_list[0].parent_x.append(end[0])
    node_list[0].parent_y.append(end[1])

    total_iter = 0
    i = 1
    pathFound = False

    potential_map = np.ones((height, length))
    potential_map[end[1]][end[0]] = 0 
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
    for _ in range(la_place_at_start):
        potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list)
    while pathFound == False:
        total_iter = total_iter + 1

        if(total_iter == iterations):
            print("Iteration limit exceeded.")
            return node_list

        # Get random point
        new_x, new_y, potential_map = random_point(start, height, length, potential_map, file_name, image, end, prob_of_choosing_start, show_every_attempted_point, show_expansion)
        

        new_node_list = try_grad_descent(potential_map, step_size, new_x, new_y, node_list)
        i = i + len(new_node_list)
        node_list.extend(new_node_list)
        len_per_iter.append(len(new_node_list))
        # HERE - Drawing image step by step
        im = Image.open(file_name)
        result = im.copy() # result image
        draw_result(image, result, start, end, node_list, potential_map, show_expansion)
        for x in range(new_x - 3, new_x + 3):
            for y in range(new_y - 3, new_y + 3):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (0, 255, 255))
        result_images.append(result)
        #TO HERE
        
        if len(new_node_list) != 0 and int(new_x) == start[0] and int(new_y) == start[1]:
            pathFound = True
        
        for _ in range(la_place_each_time):
            potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list)
    return node_list

def try_grad_descent(potential_map, step_size, new_x, new_y, node_list):
    poser = new_y
    posec = new_x
    toAppend = Node(int(posec), int(poser))    

    limit = 0
    new_node_list = []
    new_node_list.append(toAppend)
    while True:
        if limit > 1000:
            print("Couldn't find - Stuck :(")
            return []
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
        new_node_list.append(toAppend)
        limit = limit + 1


        for node in node_list:
            if(node.y-step_size-1 < poser and poser < node.y+step_size+1 and node.x-step_size-1 < posec and posec < node.x+step_size+1):
                
                print("Found from ")
                print(new_x)
                print(new_y)
                print("to")
                print(node.x)
                print(node.y)
                return new_node_list



def draw_result(image, result, start, end, node_list, potential_map, show_expansion):
    height = len(image)
    length = len(image[0])

    if show_expansion:
        for x in range(0, length):
            for y in range(0, height):
                if potential_map[y][x] == 1:
                    toPut =  0
                    result.putpixel((x, y), (toPut, toPut, toPut))
                else:
                    toPut = 250 - int(potential_map[y][x]*225)#200#150
                    result.putpixel((x, y), (toPut, toPut, toPut))

    # Draw all of the nodes in the RRT
    for node in node_list:
        #print(node)
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
    image = input("Enter the name of your image file (include .jpg, .png, etc.): ")
    file_name = image

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
    la_place_at_start = int(input('Enter the LaPlace iters before randomly selecting points: '))
    la_place_each_time = int(input('Enter the LaPlace iters after each point explored: '))
    prob_of_choosing_start = float(input('Enter the probability of choosing the start: '))
    show_every_attempted_point = ("y" == input('Enter "y" to show every attempted point in video: '))
    show_expansion = ("y" == input('Enter "y" to show every reachable point in video: '))
    output_path = input('Enter the desired file names (do not include .mp4 or .csv at the end, make it descriptive based on the parameters entered): ')
    fps = int(input('Enter the fps of video (recommended 10 for small videos, 30+ for longer): '))

    node_list = RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion)

    print("Drawing the result...")
    result = im.copy() 
    height = len(image)
    length = len(image[0])
    draw_result(image, result, start, end, node_list, potential_map=np.zeros((height, length)), show_expansion=False)
    im = Image.open(file_name)
    result_images.append(result)
    result.show()
    
    writer = imageio.get_writer(output_path + ".mp4", fps=fps)

    for image_filename in result_images:
        writer.append_data(np.array(image_filename))
    
    writer.close()

    with open(output_path + '.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(len_per_iter)


if __name__ == '__main__':
    main()
