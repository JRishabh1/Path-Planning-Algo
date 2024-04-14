# Implement RRT and RRT*

# JUST PULL THEIR CODE LATER!!!

# Installed VSCode Code Runner extension and checked "Run in Terminal"

import numpy as np # NumPy
import math # Math
import random # Random
from PIL import Image # Image
import cv2 # CV2
from timeit import default_timer as timer # Timer
import imageio
import matplotlib.pyplot as plt
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
        new_x = random.randint(1, length - 2)
        new_y = random.randint(1, height - 2)

        # If the pixel is black
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
    
def check_collision(image, x1, y1, x2, y2):
    height = len(image)
    length = len(image[0])

    # Point out of image bound
    if y1 < 0 or y2 > height-1 or x1 < 0 or x2 > length-1:
        nodeCon = False
    else:
        # check connection between two nodes
        if collision(image, x1, y1, x2, y2):
            nodeCon = False
        else:
            nodeCon = True

    return nodeCon

# RRT Algorithm
def RRT(image, node_list, potential_map, boundary, end, RRTIterations, laplaceIterations, step_size, file_name, scaleDownFactor, start_time, result_images, times, distances):

    # Height and length for image
    height = len(image)
    length = len(image[0])

    # Kernel for Laplace equation
    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])

    zero_tree = []
    zero_tree.append(Node(end[0], end[1]))

    total_iter = 0
    i = 0
    while True:
        total_iter = total_iter + 1

        if(total_iter >= RRTIterations):
            print("Iteration limit exceeded.")
            return []
            # break

        # Get random point
        # new_x, new_y = random_point(image, height, length)
        new_x, new_y = random_point(image, height, length)

        # Laplace Equation
        map = potential_map.copy()
        
        for j in range(laplaceIterations): 
            map = cv2.filter2D(map, -1, kernel)

            # Reset boundary points
            for a in range(len(zero_tree)):
                map[zero_tree[a].y][zero_tree[a].x] = 0
            for b in range(len(boundary)):
                map[boundary[b].y][boundary[b].x] = 1
            for y in range(height):
                map[y][0] = 1
                map[y][length-1] = 1
            for x in range(length):
                map[0][x] = 1
                map[height-1][x] = 1

        # Gradient Descent
        tree_x = 0
        tree_y = 0

        poser = new_y
        posec = new_x

        limit = 0
        new_node_list = []
        check2 = False
        # startOfLoop = True
        while True:
            if limit > 800:
                # print('Path not found! Maybe try providing more Laplace iterations!')
                break

            gradr = map[round(poser+1)][round(posec)] - map[round(poser-1)][round(posec)]
            gradc = map[round(poser)][round(posec+1)] - map[round(poser)][round(posec-1)]
            maggrad = math.sqrt(gradr**2 + gradc**2)

            if(maggrad != 0):
                # IF THIS LESS THAN ONE (MAYBE MAKE PARAMETER), "a", call this zero gradient
                a = step_size/maggrad # scale to pixel

                poser = poser-a*gradr
                posec = posec-a*gradc

            # If there is no gradient/no motion, throw away random point
            if(poser != new_y and posec != new_x):
                new_node_list.append(Node(posec, poser))
                zero_tree.append(Node(round(posec), round(poser)))
                check2 = True

                # Put image in video
                im = Image.open(file_name)
                w, h = im.size

                im = im.resize((round(w / scaleDownFactor), round(h / scaleDownFactor))) # just to make the image smaller
                videoResult = im.copy()

                draw_result(image, node_list, videoResult, end, step_size)
                draw_result(image, new_node_list, videoResult, end, step_size)
                for x in range(round(posec) - step_size, round(posec) + step_size):
                    for y in range(round(poser) - step_size, round(poser) + step_size):
                        if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                            videoResult.putpixel((x, y), (255, 0, 0))

                videoResult = videoResult.resize((w, h)) # make image large again
                result_images.append(videoResult)
            
            
            # DRAWING WITH THINNER LINE MAY REDUCE CLUSTERS!!!
            # Stop gradient descent
            check = False
            for node in node_list:
                if node.x - round((step_size+1)/2) < posec and posec < node.x + round((step_size+1)/2) and node.y - round((step_size+1)/2) < poser and poser < node.y + round((step_size+1)/2):
                    check = True
                    tree_x = node.x
                    tree_y = node.y
            if check == True:
                break
                
            # if(map[round(poser)][round(posec)] < 0.05):
            #     break

            limit = limit + 1

        if check2 == True:
            node_list.append(Node(new_x, new_y))
            zero_tree.append(Node(round(new_x), round(new_y)))

            middle_time = timer()
            times.append(middle_time - start_time)
            dist, _ = dist_and_angle(tree_x, tree_y, new_x, new_y)
            distances.append(dist)

        node_list.extend(new_node_list)
        

# Draw the nodes and the path
def draw_result(image, node_list, result, end, step_size):
    height = len(image)
    length = len(image[0])

    # Draw all of the nodes in the RRT
    # "Anti-aliasing"? when drawing pixel box, maybe not fill in 100% or brightness 100%?
    for node in node_list:
        round_x = round(node.x)
        round_y = round(node.y)

        if step_size == 1:
            if(0 < round_x and round_x < length - 1 and 0 < round_y and round_y < height - 1):
                result.putpixel((round_x, round_y), (255, 0, 0))
        else:
            for x in range(round_x - math.floor(step_size/2), round_x + math.floor(step_size/2)):
                for y in range(round_y - math.floor(step_size/2), round_y + math.floor(step_size/2)):
                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                        result.putpixel((x, y), (255, 0, 0))

    # Draw the end point
    for x in range(end[0] - step_size, end[0] + step_size):
        for y in range(end[1] - step_size, end[1] + step_size):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 0, 255))


def RRTLaplaceFunction(image, scaleDownFactor, end, RRTIterations, laplaceIterations, step_size, output_folder, output_path, 
               fps, output_image, output_plot, data_file, parameter_file):
    
    # Result Images for Video:
    result_images = []

    # Stuff for graph of gradient distances
    times = []
    distances = []

    
    file_name = image

    # Get image and convert to 2D array
    im = Image.open(image)
    w, h = im.size

    # Varying the resizing
    im = im.resize((round(w / scaleDownFactor), round(h / scaleDownFactor))) # just to make the image smaller

    image = np.asarray(im)

    height = len(image)
    length = len(image[0])

    # End Point
    end_comma = end.index(',')

    end_first_number = int(end[1:end_comma])
    end_second_number = int(end[end_comma+2:len(end)-1])

    end = (end_first_number, end_second_number)

    if(end_first_number < 0 or end_first_number >= length or end_second_number < 0 or end_second_number >= height):
        raise Exception("Sorry, the inputted end point is not in the image!")


    startTime = timer()


    node_list = [] # Node List!
    node_list.append(Node(end[0], end[1]))

    potential_map = np.ones((height, length)) # add 2 to both dimensions?

    # Get the black walls
    boundary = []
    for y in range(height):
        for x in range(length):
            if(image[y][x][0] == 0):
                if(end_first_number == x and end_second_number == y):
                    raise Exception("Sorry, the inputted end point is on the boundary!")

                boundary.append(Node(x, y))


    parent_array = RRT(image, node_list, potential_map, boundary, end, RRTIterations, laplaceIterations, step_size, 
                       file_name, scaleDownFactor, startTime, result_images, times, distances)

    result = im.copy() # result image
    draw_result(image, node_list, result, end, step_size)

    result = result.resize((w, h)) # make image large again
    result_images.append(result)

    
    endTime = timer()


    # Create and save the video
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

    data = [x, y, total_y]
    np.savetxt(output_folder + "/" + data_file, data, delimiter = ",")

    # Gradient Distances vs. Time
    plt.scatter(x, y)
    plt.title("Gradient Distances vs. Time")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Gradient Distances (in pixels)")

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
    
    plt.savefig(output_folder + "/" + "hist" + output_plot)

    plt.close()

    # Output parameters to txt file
    file = open(output_folder + "/" + parameter_file, "w")
    file.write("Enter the name of your image file (include .jpg, .png, etc.): " + file_name + "\n")
    file.write("Dimensions of image: " + str(w) + " pixels by " + str(h) + " pixels!" + "\n")
    file.write("Input the factor for scaling down the image (4 recommended): " + str(scaleDownFactor) + "\n")
    file.write("Dimensions of scaled image: " + str(round(w / scaleDownFactor)) + " pixels by " + str(round(h / scaleDownFactor)) + " pixels!" + "\n")
    file.write('Enter your end point (unit: pixel) in the form of "(x, y)": (' + str(end_first_number) + ', ' + str(end_second_number) + ')' + '\n')
    file.write('Enter the number of RRT iterations (10000 recommended): ' + str(RRTIterations) + '\n')
    file.write('Enter the number of Laplace equation iterations (250 recommended): ' + str(laplaceIterations) + '\n')
    file.write('Enter the step size of the RRT (3 recommended): ' + str(step_size) + "\n")
    file.write('Enter the name of the output folder: ' + output_folder + '\n')
    file.write('Enter the desired video file name (include .mp4 at the end, make it descriptive based on the parameters entered): ' + output_path + '\n')
    file.write('Enter the fps of video (recommended 10 for small videos, 30+ for longer): ' + str(fps) + '\n')
    file.write('Enter the output tree image file (include .png at the end): ' + output_image + '\n')
    file.write('Enter the output plot image file (include .png at the end): ' + output_plot + '\n')
    file.write('Enter the desired data file (include .csv at the end): ' + data_file + '\n')
    file.write('Iteration limit exceeded.' + '\n')
    file.write('Total Time: This took ' + str(round(endTime-startTime, 3)) + ' seconds!')
    file.close()



def main():
    conda = input("This is just for VSCode w/ Conda Python version, type anything here to start: ")

    # images = ['world1', 'world3', 'world4']
    # RRTIterations = [int(250), int(500)]
    # laplaceIterations = [int(100), int(200), int(300), int(400), int(500)]
    # images = ['world1']
    # RRTIterations = [int(100), int(150), int(200)]
    # laplaceIterations = [int(200)]
    images = ['world4']
    RRTIterations = [int(500)]
    laplaceIterations = [int(100), int(300), int(500)]

    scaleDownFactor = int(2)
    end = '(310, 178)'

    step_size = int(2)
    output_folder = 'mar4_videos'

    fps = int(120)

    
    number = 1
    for i in range(len(images)):
        for j in range(len(RRTIterations)):
            for k in range(len(laplaceIterations)):
                specific = '_' + images[i] + '_' + str(RRTIterations[j]) + '_' + str(laplaceIterations[k])

                output_path = 'video' + specific + '.mp4'
                output_image = 'result' + specific + '.png'
                output_plot = 'plot' + specific + '.png'
                data_file = 'data' + specific + '.csv'
                parameter_file = 'parameters' + specific + '.txt'

                RRTLaplaceFunction(images[i] + '.png', scaleDownFactor, end, RRTIterations[j], laplaceIterations[k], 
                                   step_size, output_folder, output_path, fps, output_image, output_plot, 
                                   data_file, parameter_file)
                
                number = number + 1


if __name__ == '__main__':
    main()