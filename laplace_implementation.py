import numpy as np
import math
from PIL import Image
import cv2

im = Image.open("world3.png")
im = im.resize((130,70))
image = np.asarray(im)

start = [3, 3]
end = [125, 68]
#start = [5, 5]
#end = [670, 370]

def Laplace(image, start, end):
    height = len(image)
    length = len(image[0])
    potential_map = np.zeros((height, length))
    potential_map.fill(255)
    #potential_map[start[1]][start[0]] = 0 
    potential_map[end[1]][end[0]] = 0

    boundary = []
    for y in range(height):
        for x in range(length):
            if(image[y][x][0] == 0):
                boundary.append([y,x])

    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])

    # Utilizing anthony's idea to use cv2
    for i in range(15000): # Would have to do a lot more iterations to get this to work with bigger images
        potential_map = cv2.filter2D(potential_map, -1, kernel)
        
        # potential_map[start[1]][start[0]] = 0 - THIS WAS MY MISTAKE
        potential_map[end[1]][end[0]] = 0
        for b in range(len(boundary)):
            potential_map[boundary[b][0]][boundary[b][1]] = 255
        for y in range(height):
            potential_map[y][0] = 255
            potential_map[y][length-1] = 255
        for x in range(length):
            potential_map[0][x] = 255
            potential_map[height-1][x] = 255

    draw_result(image, potential_map)

def draw_result(image, voltage):
    result = im.copy()

    gradient_descent(voltage, result)
    
    result.show()

def gradient_descent(voltage, result):
    x_coord = start[1]
    y_coord = start[0]

    iter = 20000000
    while(abs(x_coord - end[1]) > 1 or abs(y_coord - end[0]) > 1) and iter > 0:
        if(iter%50 == 0):
            print(str(x_coord) + ","+ str(y_coord)+ "," + str(iter))

        x_grad = (voltage[round(x_coord+1)][round(y_coord)] - voltage[round(x_coord-1)][round(y_coord)])/255
        y_grad = (voltage[round(x_coord)][round(y_coord+1)] - voltage[round(x_coord)][round(y_coord-1)])/255
        magnitude = math.sqrt(x_grad**2 + y_grad**2)

        if(magnitude != 0):
            x_coord = x_coord-2*x_grad/magnitude
            y_coord = y_coord-2*y_grad/magnitude
        
        result.putpixel((round(y_coord), round(x_coord)), (255,0,0))

        iter -= 1


def main():
    Laplace(image, start, end)

if __name__ == '__main__':
    main()
