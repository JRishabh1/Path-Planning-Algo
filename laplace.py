import numpy as np
from PIL import Image

# Get image and convert to 2D array
im = Image.open("world3.png")
image = np.asarray(im)

start = [5, 5]
end = [670, 370]

def averagePoints(boundary_c, voltage):
    new_voltage = np.copy(voltage)
    for i in range(len(boundary_c)):
        for j in range(len(boundary_c[0])):
            if not boundary_c[i][j] == 1:
                num = 0
                total = 0
                if i != 0:
                    num += 1
                    total += voltage[i-1][j]
                if j != 0:
                    num += 1
                    total += voltage[i][j-1]
                if i != len(boundary_c)-1:
                    num += 1
                    total += voltage[i+1][j]
                if j != len(boundary_c[0])-1:
                    num += 1
                    total += voltage[i][j+1]
                new_voltage[i][j] = total/num
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

    for i in range(5000): # todo
        voltage = averagePoints(boundary_c, voltage)
        #print(voltage)
    print(voltage)

    # given voltages, now find the path

def main():
    Laplace(image, start, end)

if __name__ == '__main__':
    main()
