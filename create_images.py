from timeit import default_timer as timer # Timer
import numpy as np
from PIL import Image, ImageDraw, ImageOps

def main():

    im = Image.new('RGB', (90, 90), (255, 255, 255))
    # im.show()
    draw = ImageDraw.Draw(im)
    draw.rectangle(xy = (0, 0, 91, 10), 
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (80, 10, 91, 91), 
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (0, 80, 91, 91), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)) 
    draw.rectangle(xy = (0, 10, 10, 91), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)),
    draw.rectangle(xy = (25, 10, 30, 55), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)),
    draw.rectangle(xy = (25, 60, 30, 80), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)),
    draw.rectangle(xy = (60, 10, 65, 30), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)),
    draw.rectangle(xy = (60, 35, 65, 80), # here
                fill = (0, 0, 0), 
                outline = (0, 0, 0)),
    # draw.rectangle(xy = (42, 10, 47, 55), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
    # draw.rectangle(xy = (42, 60, 47, 80), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
    # draw.rectangle(xy = (55, 10, 60, 30), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
    # draw.rectangle(xy = (55, 35, 60, 80), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
    # draw.rectangle(xy = (65, 10, 70, 55), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
    # draw.rectangle(xy = (65, 60, 70, 80), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
     
    # draw.rectangle(xy = (40, 40, 50, 50), 
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)) 
    # draw.rectangle(xy = (60, 60, 70, 70), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)), 
    # draw.rectangle(xy = (40, 60, 50, 70), 
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),
    # draw.rectangle(xy = (60, 40, 70, 50), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)),  
    # draw.rectangle(xy = (60, 20, 70, 30), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)), 
    # draw.rectangle(xy = (20, 60, 30, 70), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)), 
    # draw.rectangle(xy = (20, 40, 30, 50), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)), 
    # draw.rectangle(xy = (40, 20, 50, 30), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)), 
    # draw.rectangle(xy = (20, 20, 30, 30), # here
    #             fill = (0, 0, 0), 
    #             outline = (0, 0, 0)), 

    # image = np.zeros((90, 90))
    # image = image + 255
    
    # ox, oy = [], []
    # for i in range(10, 80):
    #     ox.append(i)
    #     oy.append(10.0)
    # for i in range(10, 80):
    #     ox.append(80.0)
    #     oy.append(i)
    # for i in range(10, 81):
    #     ox.append(i)
    #     oy.append(80.0)
    # for i in range(10, 81):
    #     ox.append(10.0)
    #     oy.append(i)
    # for i in range(10, 60):
    #     ox.append(40.0)
    #     oy.append(i)
    # for i in range(0, 40):
    #     ox.append(60.0)
    #     oy.append(80.0 - i)

    # for i in range(len(ox)):
    #     x = int(ox[i])
    #     y = int(oy[i])
    #     im.putpixel((x, y), (0, 0, 0))
    
    im.show()

    im.save('TestImages/few_walls.png')

    # image = np.asarray(im)

    # print(image)

    # ADD 20 to these
    # sx = 30.0  # [m]
    # sy = 30.0  # [m]
    # gx = 70.0  # [m]
    # gy = 70.0  # [m]

    # im = Image.open('metrics.png')

    # gray_im = ImageOps.grayscale(im)

    # image = np.asarray(gray_im)

    # oy, ox = np.where(image == 0)

    # print(ox)
    # print('--')
    # print(oy)
    # print('--')

    
    # im2 = Image.new('RGB', (90, 90), (255, 255, 255))

    # for i in range(len(ox)):
    #     x = int(ox[i])
    #     y = int(oy[i])
    #     im2.putpixel((x, y), (0, 0, 0))

    # im2.show()





if __name__ == '__main__':
    main()
