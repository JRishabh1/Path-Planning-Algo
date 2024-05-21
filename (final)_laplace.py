# Implement the Laplace Planner

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2 # OpenCV

from timeit import default_timer as timer # Timer

# Do gradient descent until you reach the goal
def gradient_descent(start, end, map, iterations, step_size):
    path_y = []
    path_x = []

    poser = start[1]
    posec = start[0]
    path_y.append(poser)
    path_x.append(posec)

    limit = 0
    while True:
        if limit > iterations:
            print('Path not found! Maybe try providing more Laplace iterations!')
            break

        gradr = map[round(poser+1)][round(posec)] - map[round(poser-1)][round(posec)]
        gradc = map[round(poser)][round(posec+1)] - map[round(poser)][round(posec-1)]
        maggrad = math.sqrt(gradr**2 + gradc**2)

        if(maggrad != 0):
            a = step_size/maggrad # scale to pixel
            poser = poser-a*gradr
            posec = posec-a*gradc
        
        path_y.append(poser)
        path_x.append(posec)

        if(end[1]-step_size-1 < poser and poser < end[1]+step_size+1 and end[0]-step_size-1 < posec and posec < end[0]+step_size+1):
            break

        limit = limit + 1

    return path_y, path_x

# Draw the path on the result image
def draw_path(start, end, path_x, path_y, step_size, image, result):
    # Get height and length of image
    height = len(image)
    length = len(image[0])

    # Draw the start point
    for x in range(start[0] - step_size, start[0] + step_size):
        for y in range(start[1] - step_size, start[1] + step_size):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 255, 0))

    # Draw the end point
    for x in range(end[0] - step_size, end[0] + step_size):
        for y in range(end[1] - step_size, end[1] + step_size):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 0, 255))

    # Draw the nodes drawn from the start point to the end point
    for j in range(len(path_x)):
        px = round(path_x[j])
        py = round(path_y[j])

        # print("(" + str(px) + ", " + str(py) + ")")

        if step_size == 1:
            result.putpixel((px, py), (127, 127, 127))
        else:
            for x in range(px - math.floor(step_size/2), px + math.floor(step_size/2)):
                for y in range(py - math.floor(step_size/2), py + math.floor(step_size/2)):
                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                        result.putpixel((x, y), (127, 127, 127))
            
def main():
    conda = input("This is just for VSCode using the Conda Python version, type anything here: ")

    image = 'boxes.png'

    # Get image and convert to 2D array
    im = Image.open(image)
    # im = im.resize((300, 150)) # just to make the image smaller
    image = np.asarray(im)

    height = len(image)
    length = len(image[0])

    print('Size of image: ' + str(length) + ' pixels by ' + str(height) + ' pixels!')

    start = '(15, 15)'
    end = '(75, 75)'

    start_comma = start.index(',')
    end_comma = end.index(',')

    start_first_number = int(start[1:start_comma])
    start_second_number = int(start[start_comma+2:len(start)-1])
    end_first_number = int(end[1:end_comma])
    end_second_number = int(end[end_comma+2:len(end)-1])

    start = (start_first_number, start_second_number)
    end = (end_first_number, end_second_number)

    grad_itr = 5000
    laplace_itr = 200000
    step_size = 1

    result = im.copy() # result image

    potential_map = np.ones((height, length)) # add 2 to both dimensions?
    potential_map[end[1]][end[0]] = 0 # Goal

    boundary = np.zeros((height, length))

    for y in range(height):
        for x in range(length):
            
            if(image[y][x][0] <= 10):
                if(end_first_number == x and end_second_number == y):
                    raise Exception("Sorry, the inputted end point is on the boundary!")

                boundary[y][x] = 1

    # Get the black walls
    # boundary_x = []
    # boundary_y = []
    # for y in range(height):
    #     for x in range(length):
    #         if(image[y][x][0] == 0):
    #             boundary_x.append(x)
    #             boundary_y.append(y)

    # Kernel for Laplace equation
    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])
    
    # Map Boundary
    ph, pw = potential_map.shape
    map_boundary = np.zeros((ph, pw))
    map_boundary[:, [0, -1]] = 1
    map_boundary[[0, -1], :] = 1

    # Apply the Laplace equation on the graph a number of times (I will put into another function later)
    t1 = timer()

    for i in range(laplace_itr): 
        potential_map = cv2.filter2D(potential_map, -1, kernel)

        potential_map[end[1]][end[0]] = 0
        potential_map[boundary == 1] = 1
        potential_map[map_boundary == 1] = 1

        if potential_map[start[1]][start[0]] != 1:
            print('Laplace Iterations:', i)
            break


    # plt.imshow(potential_map, cmap='gray')
    # plt.show()
        
    path_y, path_x = gradient_descent(start, end, potential_map, grad_itr, step_size)

    t2 = timer()

    print("Total Time:", t2 - t1)    


    draw_path(start, end, path_x, path_y, step_size, image, result)

    # if height < 150 or length < 150: # for tiny images
    #     result = result.resize((300, 300))
    # result = result.resize((300, 300)) # make image large again
    result = result.resize((500, 500))
    result.show()

    # im.save('tinyimage1.png')

if __name__ == '__main__':
    main()
