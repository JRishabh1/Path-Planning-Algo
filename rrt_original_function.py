# Implement RRT and RRT*

# Installed VSCode Code Runner extension and checked "Run in Terminal"

import numpy as np 
import math 
import random
from PIL import Image, ImageDraw
from timeit import default_timer as timer # Timer
import imageio
import matplotlib.pyplot as plt
import cv2
from itertools import accumulate

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

        if(image[new_y][new_x][0] >= 220):
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

        if(y_round < 0 or y_round > height-1 or x_round < 0 or x_round > length-1 or image[y_round][x_round][0] < 20):
            return True # collision
        
    return False # no collision
    
def check_collision(image, x1, y1, x2, y2, step_size):
    dist, theta = dist_and_angle(x2, y2, x1, y1)

    x = 0
    y = 0

    if dist < step_size:
        x = x1
        y = y1
    else:
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
        # # check direct connection
        # if collision(image, x, y, end[0], end[1]):
        #     directCon = False
        # else:
        #     directCon = True
        
        # check connection between two nodes
        if collision(image, x, y, x2, y2):
            nodeCon = False
        else:
            nodeCon = True

    return (x, y, False, nodeCon)
    

# RRT Algorithm
def RRT(image, node_list, start, iterations, step_size, file_name, line_list, times, distances, start_time, result_images):
    height = len(image)
    length = len(image[0])

    node_list.append(0)
    node_list[0] = Node(start[0], start[1])
    node_list[0].parent.append(node_list[0])

    total_iter = 0
    i = 1
    while True:
        total_iter = total_iter + 1

        if(total_iter >= iterations):
            print("Iteration limit exceeded.")
            break

        # Get random point
        new_x, new_y = random_point(image, height, length)

        # Find the nearest node
        nearest_ind = nearest_node(node_list, new_x, new_y)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y

        # Check connection(s)
        tx, ty, directCon, nodeCon = check_collision(image, new_x, new_y, nearest_x, nearest_y, step_size)

        check3 = True
        if(image[round(ty)][round(tx)][0] <= 20):
            check3 = False

        # Create Video
        im = Image.open(file_name)
        videoResult = im.copy()

        draw_result(image, node_list, videoResult, start)
        result_images.append(videoResult)

        if check3 == True:

            if nodeCon:
                node_list.append(i)
                node_list[i] = Node(tx, ty)
                node_list[i].parent = node_list[nearest_ind].parent.copy()
                node_list[i].parent.append(node_list[i])

                middle_time = timer()
                times.append(middle_time - start_time)
                dist, _ = dist_and_angle(nearest_x, nearest_y, tx, ty)
                distances.append(dist)

                i = i + 1

            else:
                middle_time = timer()
                times.append(middle_time - start_time)
                distances.append(0)

        


def draw_result(image, node_list, result, start):
    draw = ImageDraw.Draw(result)

    height = len(image)
    length = len(image[0])    

    # Draw the start point
    for x in range(start[0] - 5, start[0] + 5):
        for y in range(start[1] - 5, start[1] + 5):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 255, 0))

    # Draw the end point
    # shape = (end[0] - round(step_size/2), end[1] - round(step_size/2)), (end[0] + round(step_size/2), end[1] + round(step_size/2))
    # draw.ellipse(shape, fill="white", outline="green") 

    # for x in range(end[0] - 5, end[0] + 5):
    #     for y in range(end[1] - 5, end[1] + 5):
    #         if(0 < x and x < length - 1 and 0 < y and y < height - 1):
    #             result.putpixel((x, y), (0, 0, 255))

    # Draw all of the nodes/lines in the RRT
    for node in node_list:
        round_x = round(node.x)
        round_y = round(node.y)

        parent_x = round_x
        parent_y = round_y
        if len(node.parent) >= 2:
            parent_x = round(node.parent[-2].x)
            parent_y = round(node.parent[-2].y)
       
        for x in range(round_x - 3, round_x + 3):
            for y in range(round_y - 3, round_y + 3):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (255, 0, 0))

        line = ((round_x, round_y), (parent_x, parent_y))
        draw.line((line), fill=(255, 0, 0), width=2)


def RRTOriginalFunction(image, start, iterations, step_size, output_folder, output_path, fps, 
                        output_image, output_plot, data_file, parameter_file):
    
    # Result Images for Video
    result_images = []
    
    # Stuff for graph of gradient distances
    times = []
    distances = []


    file_name = image

    # Get image and convert to 2D array
    im = Image.open(image)
    image = np.asarray(im)

    height = len(image)
    length = len(image[0])

    start_comma = start.index(',')

    start_first_number = int(start[1:start_comma])
    start_second_number = int(start[start_comma+2:len(start)-1])

    start = (start_first_number, start_second_number)

    startTime = timer()


    node_list = [] # Node List!
    line_list = [] # Line List!

    parent_array = RRT(image, node_list, start, iterations, step_size, file_name, line_list, times, distances, startTime, result_images)

    result = im.copy() # result image
    draw_result(image, node_list, result, start)

    result_images.append(result)

    # result.show()


    writer = imageio.get_writer(output_folder + "/" + output_path, fps=fps)

    for image_filename in result_images:
        writer.append_data(np.array(image_filename))

    writer.close()


    # Show and save the tree image
    result.save(output_folder + "/" + output_image)
    # result.show()

    # Generate plots and save the data to .csv file
    total_distances = list(accumulate(distances))

    x = np.array(times)
    y = np.array(distances)
    total_y = np.array(total_distances)
    # y_log = np.log(y)

    data = [x, y, total_y]
    np.savetxt(output_folder + "/" + data_file, data, delimiter = ",")

    # Gradient Distances vs. Time
    plt.scatter(x, y)
    plt.title("Gradient Distances vs. Time")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Gradient Distances (in pixels)")

    # plt.ylim(0, round(step_size * 5/4))

    a = np.polyfit(x, y, 1)
    b = np.poly1d(a)

    plt.plot(x, b(x))

    plt.savefig(output_folder + "/" + "gdt" + output_plot)

    plt.close()

    # plt.show()

    # Total Gradient Distances vs. Time
    plt.scatter(x, total_y)
    plt.title("Total Gradient Distances vs. Time")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Total Gradient Distances (in pixels)")

    # plt.ylim(0, 7500)

    c = np.polyfit(x, total_y, 1)
    d = np.poly1d(c)

    plt.plot(x, d(x))

    plt.savefig(output_folder + "/" + "tgdt" + output_plot)

    plt.close()

    # plt.show()
    
    # Histogram
    plt.hist(y, bins = 30, edgecolor='black')
    plt.title("Histogram of Gradient Distances (in pixels)")
    plt.xlabel('Gradient Distances')
    plt.ylabel('Count of Gradient Distances')

    # plt.ylim(0, 250)
    
    plt.savefig(output_folder + "/" + "hist" + output_plot)

    plt.close()

    # Log Histogram
    # plt.hist(y_log, bins = 30, edgecolor='black')
    # plt.title("Histogram of (Log of) Gradient Distances (in pixels)")
    # plt.xlabel('(Log of) Gradient Distances')
    # plt.ylabel('Count of Gradient Distances')

    # plt.ylim(0, 175)
    
    # plt.savefig(output_folder + "/" + "loghist" + output_plot)

    # plt.close()


    # Output parameters to txt file
    file = open(output_folder + "/" + parameter_file, "w")
    file.write("Name of image file: " + file_name + "\n")
    file.write("Number of iterations: " + str(iterations) + "\n")
    file.write('Start Point: (' + str(start_first_number) + ', ' + str(start_second_number) + ')' + '\n')
    file.write("Step Size: " + str(step_size) + "\n")
    file.write("FPS: " + str(fps) + "\n")
    
    file.close()


def main():
    conda = input("This is just for VSCode w/ Conda Python version, type anything here to start: ")

    # images = ['world1', 'world3', 'world4']
    # RRTIterations = [int(250), int(500)]
    # laplaceIterations = [int(100), int(200), int(300), int(400), int(500)]
    # images = ['world1']
    # RRTIterations = [int(100), int(150), int(200)]
    # laplaceIterations = [int(200)]

    images = ['world3', 'world4', 'twall3']
    iterations = [int(500)]
    step_sizes = [int(20)]

    # images = ['world4']
    # iterations = [int(100)]
    # step_sizes = [int(20)]

    start = '(620, 356)'
    output_folder = 'apr8_videos'
    fps = int(30)

    
    number = 1
    for i in range(len(images)):
        for j in range(len(iterations)):
            for k in range(len(step_sizes)):
                specific = '_' + images[i] + '_itr' + str(iterations[j]) + '_ss' + str(step_sizes[k])

                output_path = 'vid' + specific + '.mp4'
                output_image = 'res' + specific + '.png'
                output_plot = 'plot' + specific + '.png'
                data_file = 'data' + specific + '.csv'
                parameter_file = 'para' + specific + '.txt'

                RRTOriginalFunction(images[i] + '.png', start, iterations[j], step_sizes[k], output_folder, output_path, fps, 
                                    output_image, output_plot, data_file, parameter_file)
                
                number = number + 1



if __name__ == '__main__':
    main()
