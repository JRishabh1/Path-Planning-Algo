import numpy as np 
import math 
import random
from PIL import Image
import cv2
import imageio
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

timesarray = []
testarray = []
laplacetimearray = []
randompointtimearray = []
graddescenttimearray = []
# Generate a random point along the edges
def random_point(start, height, length, potential_map, file_name, image, end, prob_of_choosing_start, show_every_attempted_point, show_expansion, start_time, visualization):
    potential_map_as_image = np.uint8(255 * potential_map)  
    potential_map_as_image[(potential_map_as_image == 255) | (potential_map_as_image < 254)] = 0
    potential_map_as_image[potential_map_as_image != 0] = 255
    edge = cv2.Canny(potential_map_as_image, 50, 150)
    edge_points = np.argwhere(edge > 0)
    # cv2.imshow('Edges', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(len(edge_points))
    if(len(edge_points) == 0):
        return (-1, -1, potential_map)

    len_per_iter.append(0) 


    time_per_iter.append(time.time() - start_time)


    [new_y, new_x] = edge_points[random.randint(0, len(edge_points)-1)]
    if visualization:
        if(show_every_attempted_point and image[new_y][new_x][0] == 255 and new_y < height - 2 and new_x < length - 2):
            im = Image.open(file_name)
            result = im.copy() # result image
            draw_result(image, result, start, end, node_list, potential_map, show_expansion)#draw_result(image, result, start, end, parent_x_array, parent_y_array)
            for x in range(new_x - 3, new_x + 3):
                for y in range(new_y - 3, new_y + 3):
                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                        result.putpixel((x, y), (0, 0, 255))
            result_images.append(result)
    return (new_x, new_y, potential_map)

def LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always, node_list_shape):
    potential_map = cv2.filter2D(potential_map, -1, kernel)

    # Reset boundary points
    potential_map[end[1]][end[0]] = 0 # Goal
    # potential_map[node_list] = 0
    # for node in node_list:
    #     potential_map[min(int(node.y), height - 1)][min(int(node.x), length - 1)] = 0
    # for b in range(len(boundary_y)):
    #     potential_map[boundary_y[b]][boundary_x[b]] = 1
    # for y in range(height):
    #     potential_map[y][0] = 1
    #     potential_map[y][length-1] = 1
    # for x in range(length):
    #     potential_map[0][x] = 1
    #     potential_map[height-1][x] = 1
    # node_y = np.clip(np.array([int(node.y) for node in node_list]), 0, height - 1)
    # node_x = np.clip(np.array([int(node.x) for node in node_list]), 0, length - 1)
    
    # Set the nodes to 0 in potential_map
    # potential_map[node_y, node_x] = 0

    potential_map[node_list_shape] = 0

    # Assuming boundary_x and boundary_y are numpy arrays of indices
    # Directly set the boundary points to 1 in potential_map
    potential_map[boundary_y, boundary_x] = 1
    # height_always = potential_map.shape[0]
    # length_always = potential_map.shape[1]

    # Set the first and last columns to 1
    # potential_map[:, 0] = 1
    # potential_map[:, length_always - 1] = 1

    # # Set the first and last rows to 1
    # potential_map[0, :] = 1
    # potential_map[height_always - 1, :] = 1
    potential_map[:, [0, -1]] = 1
    potential_map[[0, -1], :] = 1
    return potential_map

# RRT Algorithm
def RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion, branches_before_each_laplace, visualization):
    height = len(image)
    length = len(image[0])

    node_list.append(Node(end[0], end[1]))
    node_list[0].parent_x.append(end[0])
    node_list[0].parent_y.append(end[1])

    total_iter = 0
    i = 1
    pathFound = False

    potential_map = np.ones((height, length))
    node_list_shape = np.zeros((height, length), dtype=bool)
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
    
    laplacetime = 0
    randompointtime = 0
    graddescenttime = 0
    
    start_time = time.time()
    height_always = potential_map.shape[0]
    length_always = potential_map.shape[1]
    for _ in range(la_place_at_start):
        potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always,node_list_shape)
    laplacetime += time.time() - start_time
    while pathFound == False:
        total_iter = total_iter + 1

        if(total_iter == iterations):
            print("Iteration limit exceeded.")
            laplacetimearray.append(laplacetime)
            randompointtimearray.append(randompointtime)
            graddescenttimearray.append(graddescenttime)
            return node_list

        # Get random point
        for _ in range(branches_before_each_laplace):
            newtime = time.time()
            new_x, new_y, potential_map = random_point(start, height, length, potential_map, file_name, image, end, prob_of_choosing_start, show_every_attempted_point, show_expansion, start_time, visualization)
            randompointtime += time.time() - newtime
            if (new_x == -1):
                laplacetimearray.append(laplacetime)
                randompointtimearray.append(randompointtime)
                graddescenttimearray.append(graddescenttime)
                return node_list
            
            othertime = time.time()
            new_node_list = try_grad_descent(potential_map, step_size, new_x, new_y, node_list)
            for node in new_node_list:
                node_list_shape[int(node.y), int(node.x)] = True
            graddescenttime += time.time() - othertime
            i = i + len(new_node_list)
            node_list.extend(new_node_list)
            if visualization:
                len_per_iter.append(len(new_node_list))
                time_per_iter.append(time.time() - start_time)
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
            
            # if len(new_node_list) != 0 and int(new_x) == start[0] and int(new_y) == start[1]:
            #     pathFound = True
        sometime = time.time()
        for _ in range(la_place_each_time):
            potential_map = LaPlace_average(potential_map, kernel, height, length, boundary_x, boundary_y, end, node_list, height_always, length_always, node_list_shape)
        laplacetime += time.time() - sometime
    # return node_list

def try_grad_descent(potential_map, step_size, new_x, new_y, node_list):
    poser = new_y
    posec = new_x
    toAppend = Node(int(posec), int(poser))    

    limit = 0
    new_node_list = []
    new_node_list.append(toAppend)
    while True:
        if limit > 1000:
            # print("Couldn't find - Stuck :(") - if there is a meaningful gradient from the first point, should never hit this
            return []
        
        # gradr = potential_map[round(poser+1)][round(posec)] - potential_map[round(poser-1)][round(posec)]
        # gradc = potential_map[round(poser)][round(posec+1)] - potential_map[round(poser)][round(posec-1)]
        if 0 < poser < potential_map.shape[0]-1 and 0 < posec < potential_map.shape[1]-1:
            gradr = potential_map[int(poser+1)][int(posec)] - potential_map[int(poser-1)][int(posec)]
            gradc = potential_map[int(poser)][int(posec+1)] - potential_map[int(poser)][int(posec-1)]
        else:
            # Avoid calculations that would go out of bounds
            return []
        maggrad = math.sqrt(gradr**2 + gradc**2)

        if(maggrad != 0):
            a = step_size/maggrad
            poser = poser-a*gradr
            posec = posec-a*gradc
        toAppend = Node(int(posec), int(poser))
        new_node_list.append(toAppend)
        limit = limit + 1


        for node in node_list:
            if(node.y-step_size-1 < poser and poser < node.y+step_size+1 and node.x-step_size-1 < posec and posec < node.x+step_size+1):      
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
    

                
def running_hybridization(image, start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branches_before_each_laplace, visualization):
    file_name = image
    timer = time.time()
    # Get image and convert to 2D array
    im = Image.open(image)
    image = np.asarray(im)

    start_comma = start.index(',')
    end_comma = end.index(',')

    start_first_number = int(start[1:start_comma])
    start_second_number = int(start[start_comma+2:len(start)-1])
    end_first_number = int(end[1:end_comma])
    end_second_number = int(end[end_comma+2:len(end)-1])

    start = (start_first_number, start_second_number)
    end = (end_first_number, end_second_number)
    node_list = RRT(image, start, end, iterations, step_size, file_name, prob_of_choosing_start, la_place_at_start, la_place_each_time, show_every_attempted_point, show_expansion, branches_before_each_laplace, visualization)

    if visualization:
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
            csv_writer.writerow(time_per_iter)
    end_time = time.time()
    timesarray.append(end_time - timer)
    testarray.append(output_path)
    print(output_path, " took ", str(end_time - timer), " seconds with ", laplacetimearray[len(laplacetimearray) - 1], " laplace time ",randompointtimearray[len(randompointtimearray) - 1], " random poin with image segtime ", graddescenttimearray[len(graddescenttimearray) - 1], " grad descent time")


def csv_to_graph(file):
    data = pd.read_csv(file + ".csv", header=None)

    values = data.iloc[0].values
    time = data.iloc[1].values

    # Compute prefix sum
    prefix_sum = values.cumsum()

    # Plot of each point
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(values, marker='o', linestyle=' ')
    plt.title('Original Values')
    plt.xlabel('Iterations')
    plt.ylabel('Length of Branch Added')

    # Plot using prefix sum
    plt.subplot(2, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(prefix_sum, marker='o', linestyle='-')
    plt.title('Cumulative Sum')
    plt.xlabel('Iterations')
    plt.ylabel('Prefix Sum of Length of Branch Added')

    plt.subplot(2, 2, 3)
    plt.hist(values, bins=30, alpha=0.5, color='steelblue', edgecolor='black')
    plt.title('Histogram of Branch Lengths')

    plt.xlabel('Branch Length')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 4)
    plt.plot(time, prefix_sum, marker='o', linestyle='') 

    # Customize the plot
    plt.title('Time vs Prefix Sum')
    plt.xlabel('Time')
    plt.ylabel('Prefix Sum')

    plt.tight_layout()
    plt.savefig(file + '.png', dpi=300, bbox_inches='tight')  


# TESTS BEING RUN:

images_to_test = ["world1", "world2", "world3", "world4", "t_shape", "t_shape_other_way"]
start = "(1, 1)"
end = "(650, 350)"
iterations = 1000
step_size = 3
laplace_iters_to_test = [50, 100, 200, 300, 500]
branches_before_each_laplace = [1, 5, 10]
prob_of_choosing_start = 0
show_every_attempted_point = "y"
show_expansion = "y"
fps = 20

for i in range(len(images_to_test)):
    image = images_to_test[i]
    for j in range(len(laplace_iters_to_test)):
        for k in range(len(branches_before_each_laplace)):
            la_place_at_start = laplace_iters_to_test[j]
            la_place_each_time = laplace_iters_to_test[j]
            branch_each_time = branches_before_each_laplace[k]
            len_per_iter = []
            time_per_iter = []
            node_list = []
            result_images = []
            output_path = str(image) + "_" + str(la_place_each_time) + "laplace_" + str(branch_each_time) + "branches_each_iter"
            visualition = False
            running_hybridization(image + ".png", start, end, iterations, step_size, la_place_at_start, la_place_each_time, prob_of_choosing_start, show_every_attempted_point, show_expansion, output_path, fps, branch_each_time, visualition)
            if visualition:
                csv_to_graph(output_path)
            # print("done with ")
            # print(output_path)



with open('april_24_times_compiled.csv', 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Test Name'] + testarray)
    csv_writer.writerow(['Total Time'] + timesarray)
    csv_writer.writerow(['LaPlace Time'] + laplacetimearray)
    csv_writer.writerow(['Random Point from Image segmentation Time'] + randompointtimearray)
    csv_writer.writerow(['Grad Descent Time'] + graddescenttimearray)
