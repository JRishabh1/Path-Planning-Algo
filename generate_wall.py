from timeit import default_timer as timer # Timer
import numpy as np
from PIL import Image, ImageDraw

def main():

    im = Image.new('RGB', (90, 90), (255, 255, 255))
    # im.show()
    draw = ImageDraw.Draw(im)
    draw.rectangle(xy = (10, 10, 80, 10), 
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (80, 10, 80, 80), 
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (10, 80, 80, 80), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (10, 10, 10, 80), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (40, 10, 40, 60), 
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (60, 41, 60, 80), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    
    im.show()

    im.save('metrics.png')

    # image = np.asarray(im)

    # print(image)

    # ADD 20 to these
    # sx = 10.0  # [m]
    # sy = 10.0  # [m]
    # gx = 50.0  # [m]
    # gy = 50.0  # [m]



if __name__ == '__main__':
    main()
