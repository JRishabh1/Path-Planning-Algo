# Implement the Laplace Planner

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2 # OpenCV

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

        for x in range(px - math.floor(step_size/2), px + math.floor(step_size/2)):
            for y in range(py - math.floor(step_size/2), py + math.floor(step_size/2)):
                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                    result.putpixel((x, y), (127, 127, 127))
            
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

    grad_itr = int(input('Enter the number of gradient descent iterations (1000 recommended): '))
    laplace_itr = int(input('Enter the number of Laplace iterations (10000-100000 recommended): '))
    step_size = int(input('Enter the gradient descent step size (2-10 recommended): '))

    result = im.copy() # result image

    height = len(image)
    length = len(image[0])

    print('Size of image: ' + str(height) + ' pixels by ' + str(length) + ' pixels!')

    potential_map = np.ones((height, length)) # add 2 to both dimensions?
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

    # Apply the Laplace equation on the graph a number of times (I will put into another function later)
    for i in range(laplace_itr): 
        potential_map = cv2.filter2D(potential_map, -1, kernel)

        # Reset boundary points
        potential_map[end[1]][end[0]] = 0 # Goal
        for b in range(len(boundary_y)):
            potential_map[boundary_y[b]][boundary_x[b]] = 1
        for y in range(height):
            potential_map[y][0] = 1
            potential_map[y][length-1] = 1
        for x in range(length):
            potential_map[0][x] = 1
            potential_map[height-1][x] = 1

    # plt.imshow(potential_map, cmap='gray')
    # plt.show()
        
    path_y, path_x = gradient_descent(start, end, potential_map, grad_itr, step_size)
    draw_path(start, end, path_x, path_y, step_size, image, result)

    if height < 150 or length < 150:
        result = result.resize((300, 300))
    result.show()

    # im.save('tinyimage1.png')

if __name__ == '__main__':
    main()
