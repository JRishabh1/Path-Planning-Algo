import numpy as np
import math
from PIL import Image

# Get image and convert to 2D array
im = Image.open("world3.png")
image = np.asarray(im)

start = [5, 5]
end = [670, 370]

def averagePoints(boundary_c, voltage):
    new_voltage = np.copy(voltage)
    for i in range(1, len(boundary_c) - 1):
        for j in range(1, len(boundary_c[0]) - 1):
            if not boundary_c[i][j] == 1:
                # num = 0.0
                # total = 0.0
                # if i != 0:
                #     num += 1
                #     total += voltage[i-1][j]
                # if j != 0:
                #     num += 1
                #     total += voltage[i][j-1]
                # if i != len(boundary_c)-1:
                #     num += 1
                #     total += voltage[i+1][j]
                # if j != len(boundary_c[0])-1:
                #     num += 1
                #     total += voltage[i][j+1]
                new_voltage[i][j] = 0.25 * (voltage[i][j+1] +voltage[i+1][j] +  voltage[i-1][j] + voltage[i][j-1])
    return new_voltage

# Laplace Algorithm
def Laplace(image, start, end):
    height = len(image)
    length = len(image[0])

    boundary_c = np.zeros((height, length), dtype=int)
    voltage = np.zeros((height, length), dtype=float)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j][0] != 255:
                boundary_c[i][j] = 1
                voltage[i][j] = 255
            elif (start[0] == i and start[1] == j) or (end[0] == i and end[1] == j):
                boundary_c[i][j] = 1
                voltage[i][j] = 0
            else:
                boundary_c[i][j] = 0
                voltage[i][j] = 0
            if(i == 0 or i == len(image) - 1) or (j == 0 or j == len(image[0] - 1)):
                boundary_c[i][j] = 1
                voltage[i][j] = 255
    for i in range(5000): # todo
        voltage = averagePoints(boundary_c, voltage)
        if(i%100 == 0):
            print(i)
        #print(voltage)
    print(voltage)
    draw_result(image, voltage)

    # given voltages, now find the path
def draw_result(image, voltage):
    height = len(image)
    length = len(image[0])
    result = im.copy()

    result.putpixel((5,5), (255,0,0))
    result.putpixel((670, 370), (0,255,0))
    gradient_descent(voltage, result)
    #for x in range(height):
     #   for y in range(length):
            # WRITEEEE
      #      result.putpixel((y,x), (0,0,round(voltage[x][y])))
    
    result.show()

def gradient_descent(voltage, result):
    x_coord = start[1]
    y_coord = start[0]

    iter = 200000
    while(abs(x_coord - end[1]) > 1 or abs(y_coord - end[0]) > 1) and iter > 0:
        if(iter%50 == 0):
            print(str(x_coord) + ","+ str(y_coord)+ "," + str(iter))
        grad_x = (voltage[round(x_coord+1)][round(y_coord)] - voltage[round(x_coord-1)][round(y_coord)])/255
        grad_y = (voltage[round(x_coord)][round(y_coord+1)] - voltage[round(x_coord)][round(y_coord-1)])/255
        magnitude = math.sqrt(grad_x**2 + grad_y**2)

        x_coord = x_coord - 2/magnitude*grad_x
        y_coord = y_coord - 2/magnitude*grad_y

        result.putpixel((round(y_coord), round(x_coord)), (255,0,0)) 
        iter -= 1


def main():
    Laplace(image, start, end)

if __name__ == '__main__':
    main()
