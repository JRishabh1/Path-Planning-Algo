# Implement the Laplace Planner

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
# import OpenCV???

# Each iteration takes a long time
def Laplace(image, potential_map):
    h = len(image)
    l = len(image[0])

    # create a new map and return that
    result = potential_map.copy()

    for y in range(len(result)):
        for x in range(len(result[0])):
            if x != 0 and x != l-1 and y != 0 and y != h-1 and image[y][x][0] == 255 and potential_map[y][x] != 0:
                result[y][x] = potential_map[y-1][x] + potential_map[y+1][x] + potential_map[y][x-1] + potential_map[y][x+1]
                result[y][x] = result[y][x] / 4

    return result

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
            print('Iteration limit exceeded!')
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
def draw_path(start, end, path_x, path_y, image, result):
    # Get height and length of image
    height = len(image)
    length = len(image[0])

    # Draw the start point
    for x in range(start[0] - 1, start[0] + 1):
        for y in range(start[1] - 1, start[1] + 1):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 255, 0))

    # Draw the end point
    for x in range(end[0] - 1, end[0] + 1):
        for y in range(end[1] - 1, end[1] + 1):
            if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                result.putpixel((x, y), (0, 0, 255))

    # Draw the nodes drawn from the start point to the end point
    for j in range(len(path_x)):
        x = math.floor(path_x[j])
        y = math.floor(path_y[j])

        if(0 < x and x < length - 1 and 0 < y and y < height - 1):
            result.putpixel((x, y), (127, 127, 127))
            
def main():
    conda = input("This is just for VSCode using the Conda Python version, type anything here: ")
    image = input("Enter the name of your image file (include .jpg, .png, etc.): ")

    # Get image 
    im = Image.open(image)

    # Draw Obstacles

    # Convert image to 2D array
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

    grad_itr = int(input('Enter the number of gradient descent iterations (10000 recommended): '))
    laplace_itr = int(input('Enter the number of Laplace iterations (5000 recommended): '))
    step_size = int(input('Enter the gradient descent step size (2 recommended): '))

    result = im.copy() # result image

    height = len(image)
    length = len(image[0])

    print('Size of image: ' + str(height) + ' pixels by ' + str(length) + ' pixels!')

    potential_map = np.ones((height, length)) # add 2 to both dimensions?
    potential_map[end[1]][end[0]] = 0 # Goal

    # Apply the Laplace equation on the graph a number of times
    for i in range(laplace_itr): 
        potential_map = Laplace(image, potential_map)

    plt.imshow(potential_map, cmap='gray')
    plt.show()
        
    path_y, path_x = gradient_descent(start, end, potential_map, grad_itr, step_size)
    draw_path(start, end, path_x, path_y, image, result)

    result = result.resize((300, 300))
    result.show()

    # im.save('tinyimage1.png')

if __name__ == '__main__':
    main()
