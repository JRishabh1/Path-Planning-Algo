# Implement RRT and RRT*

# JUST PULL THEIR CODE LATER!!!

# Installed VSCode Code Runner extension and checked "Run in Terminal"

import cupy as cp
import numpy as np # NumPy
import math # Math
import random # Random
from PIL import Image, ImageOps # Image
import cv2 # CV2
from timeit import default_timer as timer # Timer
import imageio
import matplotlib.pyplot as plt
from itertools import accumulate
from cupyx.scipy.signal import convolve2d

# Look at CuPy and PyTorch


# Ideas for GPU Acceleration:
# 1. Use CuPy library (like NumPy/SciPy but for GPUs)
    # a. This is not only faster for convolutions, but also for other NumPy stuff I think
    # b. https://www.geeksforgeeks.org/python-cupy/
# 2. Nvidia cuDNN library
# 3. PyTorch


class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = []
        self.point = False


# Generate a random point in the maze
def random_point(image, height, length, edges):
    check = True
    new_x = -1
    new_y = -1

    while check:
        if len(edges) == 0:
            new_x = random.randint(1, length - 2)
            new_y = random.randint(1, height - 2)

        else:
            node = random.choice(edges)
            # new_x = node.x
            # new_y = node.y
            new_x = node[1] + 4
            new_y = node[0] + 4
            

            if new_x <= 3 or new_x >= length-3:
                if new_y <= 3 or new_y >= height-3:
                    new_x = random.randint(1, length - 2)
                    new_y = random.randint(1, height - 2)

        # If the pixel is black
        check = False
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i + new_y >= 0 and i + new_y < height and j + new_x >= 0 and j + new_x < length:
                    if(image[i+new_y][j+new_x][0] <= 10):
                        check = True
    
    return (new_x, new_y)

# Return the distance and angle between the new point and nearest node
def dist_and_angle(x1, y1, x2, y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2 - y1, x2 - x1)
    return (dist, angle)

# RRT Algorithm
def RRT(image, node_list, potential_map, boundary, end, RRTIterations, laplaceIterations, step_size, file_name, scaleDownFactor, start_time, result_images, times, distances, bpl, no_draw, w, h):

    # Constants
    ph, pw = potential_map.shape
    map_boundary = np.zeros((ph, pw))
    for y in range(ph):
        map_boundary[y][0] = 1
        map_boundary[y][pw-1] = 1
    for x in range(pw):
        map_boundary[0][x] = 1
        map_boundary[ph-1][x] = 1

    # Times for Each Operation
    laplace_time = 0
    edge_detection_time = 0
    gradient_descent_time = 0
    drawing_time = 0

    # Height and length for image
    height = len(image)
    length = len(image[0])

    # Kernel for Laplace equation
    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])

    # Zero Tree
    zero_tree = np.zeros((ph, pw))
    zero_tree[end[1]][end[0]] = 1


    total_iter = 0
    while True:

        # Laplace Equation
        m1 = timer()

        map = potential_map.copy()
        
        for j in range(laplaceIterations): 
            # use pytorch or cupy for this
            map = cp.asarray(map)
            map = convolve2d(map, kernel, 'same')
            map = cp.asnumpy(map)

            map[zero_tree == 1] = 0
            map[boundary == 1] = 1
            map[map_boundary == 1] = 1

        m2 = timer()

        laplace_time += (m2 - m1)


        # Edge Detection
        m1 = timer()

        new_map = map.copy()
        new_map[new_map != 1] = 50
        new_map[new_map == 1] = 0
        new_map = np.uint8(new_map)
       
        # Go from PIL to CV2
        edge = cv2.Canny(new_map, 50, 150) 
        
        # Go from CV2 to PIL
        edge = Image.fromarray(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))

        # Convert to grayscale image
        new_edge = ImageOps.grayscale(edge)

        # Crop image to avoid boundary edges
        ewidth, eheight = new_edge.size
        new_edge = new_edge.crop((2, 2, ewidth-3, eheight-3)) 

        # Get array of edges
        edgeArray = np.asarray(new_edge)
        edges = np.argwhere(edgeArray >= 240)

        m2 = timer()

        edge_detection_time += (m2 - m1)


        # Drawing
        if no_draw == False:
            m1 = timer()

            test_im3 = Image.open(file_name)
            w3, h3 = test_im3.size

            test_im3 = test_im3.resize((round(w3 / scaleDownFactor), round(h3 / scaleDownFactor))) # just to make the image smaller
            testResult3 = test_im3.copy()

            tw3, th3 = testResult3.size
            # i have to iterate over every pixel in the image for this
            # Is there a faster way than this?
            for ti in range(tw3):
                for tj in range(th3):
                    tgs3 = 0 # black
                    if(map[tj][ti] != 1): # NumPy array of potential
                        tgs3 = max(255 - round(map[tj][ti] * 255), 50)

                    # YOU CAN CREATE PIL IMAGE FROM NUMPY ARRAY!!!
                    testResult3.putpixel((ti, tj), (tgs3, tgs3, tgs3)) # PIL Image


            videoResult = testResult3.copy()
            bh, bw = boundary.shape
            for by in range(bh):
                for bx in range(bw):
                    if(boundary[by][bx] == 1):
                        videoResult.putpixel((bx, by), (0, 0, 127))
        
            draw_result(image, node_list, videoResult, end, step_size)

            m2 = timer()

            drawing_time += (m2 - m1)


        # Gradient Descent
        gm1 = timer()

        temp_drawing_time = 0

        for stuff in range(bpl):
            # Total Iterations Count
            total_iter = total_iter + 1

            if(total_iter >= RRTIterations):
                print("Iteration limit exceeded.")
                return [laplace_time, edge_detection_time, gradient_descent_time, drawing_time]
                # break

            # Get random point
            new_x, new_y = random_point(image, height, length, edges)


            tree_x = 0
            tree_y = 0

            poser = new_y
            posec = new_x

            limit = 0
            new_node_list = []
            check2 = False

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

                # If there is no gradient/no motion, throw away random points
                if(poser != new_y and posec != new_x):
                    new_node_list.append(Node(posec, poser))

                    zero_tree[round(poser)][round(posec)] = 1
                    check2 = True

                    if no_draw == False:
                        dm1 = timer()

                        # Put image in video
                        vResult = videoResult.copy()
                        
                        draw_result(image, new_node_list, vResult, end, step_size)
                        for x in range(round(posec) - 2, round(posec) + 2):
                            for y in range(round(poser) - 2, round(poser) + 2):
                                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                                    vResult.putpixel((x, y), (255, 0, 0))

                        vResult = vResult.resize((w, h)) # make image large again
                        result_images.append(vResult)

                        dm2 = timer()
                        temp_drawing_time += (dm2 - dm1)
                
                # Stop gradient descent
                break_check = False
                for sy in range(0, 2):
                    for sx in range(0, 2):
                        if map[math.floor(poser)+sy][math.floor(posec)+sx] == 0:
                            break_check = True
                if break_check == True:
                    break

                limit = limit + 1

            
            if check2 == True:
                pt = Node(new_x, new_y)
                pt.point = True
                node_list.append(pt)
                zero_tree[round(new_y)][round(new_x)] = 1

                middle_time = timer()
                times.append(middle_time - start_time)
                dist, _ = dist_and_angle(tree_x, tree_y, new_x, new_y)
                distances.append(dist)
            else:
                middle_time = timer()
                times.append(middle_time - start_time)
                distances.append(0)

            node_list.extend(new_node_list)

        gm2 = timer()

        gradient_descent_time += (gm2 - gm1)
        gradient_descent_time -= temp_drawing_time
        drawing_time += temp_drawing_time

        
        

# Draw the nodes and the path
def draw_result(image, node_list, result, end, step_size):
    height = len(image)
    length = len(image[0])

    # Draw all of the nodes in the RRT
    # "Anti-aliasing"? when drawing pixel box, maybe not fill in 100% or brightness 100%?
    for node in node_list:
        round_x = round(node.x)
        round_y = round(node.y)

        if node.point == False:
            if(0 < round_x and round_x < length - 1 and 0 < round_y and round_y < height - 1):
                result.putpixel((round_x, round_y), (255, 0, 0))
        else:
            for x in range(round_x - 2, round_x + 2):
                for y in range(round_y - 2, round_y + 2):
                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                        result.putpixel((x, y), (255, 0, 0))

    # Draw the end point
    for x in range(end[0] - 3, end[0] + 3):
        for y in range(end[1] - 3, end[1] + 3):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 0, 255))

    # print(height, length, end)


def RRTLaplaceFunction(image, scaleDownFactor, end, RRTIterations, laplaceIterations, step_size, output_folder, output_path, 
               fps, output_image, output_plot, data_file, parameter_file, time_file, bpl, no_draw):
    
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
    boundary = np.zeros((height, length))

    for y in range(height):
        for x in range(length):
            
            if(image[y][x][0] <= 10):
                if(end_first_number == x and end_second_number == y):
                    raise Exception("Sorry, the inputted end point is on the boundary!")

                boundary[y][x] = 1


    lt, edt, gdt, dt = RRT(image, node_list, potential_map, boundary, end, RRTIterations, laplaceIterations, step_size, 
                       file_name, scaleDownFactor, startTime, result_images, times, distances, bpl, no_draw, w, h)
    

    endTime = timer()

    print(lt, edt, gdt, dt)

    result = im.copy() # result image
    draw_result(image, node_list, result, end, step_size)

    result = result.resize((w, h)) # make image large again


    if no_draw == False:
        # Create and save the video
        result_images.append(result)

        writer = imageio.get_writer(output_folder + "/" + output_path, fps=fps) # removed fps?

        for image_filename in result_images:
            writer.append_data(np.array(image_filename))

        writer.close()

    # Show and save the tree image
    result.save(output_folder + "/" + output_image)

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
    file.write('Enter your start point (unit: pixel) in the form of "(x, y)": (' + str(end_first_number) + ', ' + str(end_second_number) + ')' + '\n')
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

    # Output times to txt file
    file = open(output_folder + "/" + time_file, "w")
    file.write("Laplace Time: " + str(lt) + "\n")
    file.write("Edge Detection Time: " + str(edt) + "\n")
    file.write("Gradient Descent Time: " + str(gdt) + "\n")
    file.write("Drawing Time: " + str(dt) + "\n")
    file.close()



def main():
    conda = input("This is just for VSCode w/ Conda Python version, type anything here to start: ")

    images = ['world4']
    RRTIterations = [int(30)]
    laplaceIterations = [int(200)]

    scaleDownFactor = int(2)
    end = '(310, 178)'

    step_size = int(1)
    output_folder = 'apr23_videos'

    fps = int(120)

    bpl = [int(1)]

    no_draw = False

    
    number = 1
    for i in range(len(images)):
        for j in range(len(RRTIterations)):
            for k in range(len(laplaceIterations)):
                for l in range(len(bpl)):
                    specific = '_ye_' + images[i] + '_rrt' + str(RRTIterations[j]) + '_lap' + str(laplaceIterations[k]) + '_bql' + str(bpl[l])

                    output_path = 'vid' + specific + '.mp4'
                    output_image = 'res' + specific + '.png'
                    output_plot = 'plot' + specific + '.png'
                    data_file = 'data' + specific + '.csv'
                    parameter_file = 'param' + specific + '.txt'
                    time_file = 'time' + specific + '.txt'

                    RRTLaplaceFunction(images[i] + '.png', scaleDownFactor, end, RRTIterations[j], laplaceIterations[k], 
                                    step_size, output_folder, output_path, fps, output_image, output_plot, 
                                    data_file, parameter_file, time_file, bpl[l], no_draw)
                    
                    number = number + 1


if __name__ == '__main__':
    main()