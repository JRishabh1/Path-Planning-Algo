# Implement the Laplace Planner

import numpy as np
from PIL import Image

# Each iteration takes a long time
def Laplace(image, potential_map):
    h = len(image)
    l = len(image[0])

    for y in range(len(potential_map)):
        for x in range(len(potential_map[0])):
            # What should I do about the image boundaries?
            if x != 0 and x != l-1 and y != 0 and y != h-1 and image[y][x][0] == 255 and potential_map[y][x] != 0:
                potential_map[y][x] = potential_map[y-1][x] + potential_map[y+1][x] + potential_map[y][x-1] + potential_map[y][x+1]
                potential_map[y][x] = potential_map[y][x] / 4

# Do gradient descent until you reach the goal
def gradient_descent(start, potential_map):
    path = []

    # Do the stuff like from the slides!

    return path

# Draw the path on the result image
def draw_path(result, path):
    
    # Implement later.

            
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

    iterations = int(input('Enter the number of iterations (10 recommended): '))

    result = im.copy() # result image

    height = len(image)
    length = len(image[0])

    potential_map = np.ones((height, length))
    potential_map[end[1]][end[0]] = 0 # Goal

    # print(potential_map)

    # Apply the Laplace equation on the graph a number of times
    for i in range(iterations): 
        Laplace(image, potential_map)

    # print('\n')
    # print(potential_map)
        
    path = gradient_descent(start, potential_map)
    draw_path(result, path)

    result.show()

if __name__ == '__main__':
    main()
