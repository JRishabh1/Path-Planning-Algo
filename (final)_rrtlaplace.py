import numpy as np # NumPy
import math # Math
import random # Random
from PIL import Image, ImageOps # Image
import cv2 # CV2
from timeit import default_timer as timer # Timer
import imageio
import matplotlib.pyplot as plt
from itertools import accumulate
import csv # save to CSV file
from datetime import datetime


# Node Class
class Node:
    """Class to store the RRT graph"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = []
        self.point = False


# Generate a random point in the maze
def random_point(image, height, length, edges, extra_boundary):
    check = True
    new_x = -1
    new_y = -1

    while check:
        # If there are no more edges to choose from, just choose a random point on the map
        if len(edges) == 0: 
            new_x = random.randint(1, length - 2)
            new_y = random.randint(1, height - 2)

        # Choose a random point from the (white) edges
        else:
            node = random.choice(edges)
            new_x = node[1]
            new_y = node[0]


        # If the pixel is not on the boundary, stop and return this random point!
        if extra_boundary[new_y][new_x] == 0:
            check = False

    return (new_x, new_y)


# Return the distance and angle between two points!
def dist_and_angle(x1, y1, x2, y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2 - y1, x2 - x1)
    return (dist, angle)


# RRT-Laplace Algorithm
def RRT(image, node_list, potential_map, boundary, extra_boundary, start, end, RRTIterations, laplaceIterations, step_size, file_name, scaleDownFactor, start_time, result_images, times, distances, bpl, no_draw, w, h, result_edges, show_animation):

    # Constants to check if all the map has been explored
    map_explored = False
    exploring_iterations = 0

    ph, pw = potential_map.shape


    # Times for Each Operation
    laplace_time = 0
    edge_detection_time = 0
    gradient_descent_time = 0
    drawing_time = 0


    # Height and length for image
    height = len(image)
    length = len(image[0])

    print("Height: " + str(height))
    print("Length: " + str(length))


    # Kernel for Laplace equation
    kernel = np.array([[0.00, 0.25, 0.00],
                       [0.25, 0.00, 0.25],
                       [0.00, 0.25, 0.00]])
    

    # Zero Tree
    zero_tree = np.zeros((ph, pw))
    zero_tree[start[1]][start[0]] = 1

    # The boundaries/edges of the image
    map_boundary = np.zeros((ph, pw))
    map_boundary[:, [0, -1]] = 1
    map_boundary[[0, -1], :] = 1


    total_iter = 0 # counts the total number of RRT iterations
    while True:

        # Laplace Equation
        temp_animation_time = 0

        m1 = timer()

        map = potential_map.copy()
        
        for _ in range(laplaceIterations): 
            # use pytorch or cupy for this
            map = cv2.filter2D(map, -1, kernel)

            map[zero_tree == 1] = 0
            map[boundary == 1] = 1
            map[map_boundary == 1] = 1

            
            # Draw new stuff
            pm1 = timer()

            if show_animation == True:
                row_indices, col_indices = np.nonzero(map != 1)
                row_boundaries, col_boundaries = np.nonzero(zero_tree)
                plt.plot(col_indices, row_indices, "xc")
                plt.plot(col_boundaries, row_boundaries, ".r")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                plt.pause(1e-5)

            pm2 = timer()

            temp_animation_time += (pm2 - pm1)


            # Stop when end point is covered by gray area
            if map[end[1]][end[0]] != 1:
                print("End point covered!")
                # LAPLACE TIME INACCURATE BELOW, FIX LATER, ALSO DRAWING TIME AND EXPLORING ITERATIONS
                new_x, new_y = end[1], end[0]

                tree_x = 0
                tree_y = 0

                poser = end[1]
                posec = end[0]

                limit = 0
                new_node_list = []
                check2 = False # check to see if there was a gradient found between the random point and zero tree

                while True:
                    # this is just for stopping the program if there's some bug in Gradient Descent
                    if limit > 800:
                        break

                    # Taking a step in Gradient Descent
                    gradr = map[round(poser+1)][round(posec)] - map[round(poser-1)][round(posec)]
                    gradc = map[round(poser)][round(posec+1)] - map[round(poser)][round(posec-1)]
                    maggrad = math.sqrt(gradr**2 + gradc**2)

                    if(maggrad != 0):
                        a = step_size/maggrad # scale to pixel

                        poser = poser-a*gradr
                        posec = posec-a*gradc


                    # If there is no gradient/no motion, throw away this chosen random point!
                    if(poser != new_y and posec != new_x):
                        # Creating the new red node
                        new_node_list.append(Node(posec, poser))
                        zero_tree[round(poser)][round(posec)] = 1


                        # temp2_drawing_time = 0

                        # # for matplotlib!!!
                        # if show_animation:
                        #     pm1 = timer()

                        #     plt.plot(round(posec), round(poser), "xc")
                        #     plt.pause(0.00001)

                        #     pm2 = timer()

                        # temp2_drawing_time += (pm2 - pm1)


                        check2 = True


                        # Drawing stuff for Gradient Descent
                        if no_draw == False:
                            dm1 = timer()

                            # Put image in video
                            vResult = videoResult.copy()
                            
                            draw_result(image, new_node_list, vResult, start, end)
                            for x in range(round(posec) - 2, round(posec) + 2):
                                for y in range(round(poser) - 2, round(poser) + 2):
                                    if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                                        vResult.putpixel((x, y), (255, 0, 0))

                            vResult = vResult.resize((w, h)) # make image large again
                            result_images.append(vResult)

                            dm2 = timer()
                            temp_drawing_time += (dm2 - dm1)
                    
                    # # Stop when end point is covered by gray area
                    # if map[end[1]][end[0]] != 1:
                    #     print("End point covered!")
                    #     return [laplace_time, edge_detection_time, gradient_descent_time, drawing_time, exploring_iterations]

                    # Stopping Gradient Descent once it has reached the Zero Tree (all the red nodes)
                    break_check = False
                    for sy in range(0, 2):
                        for sx in range(0, 2):
                            if map[math.floor(poser)+sy][math.floor(posec)+sx] == 0:
                                break_check = True
                    if break_check == True:
                        break

                    limit = limit + 1

                            # Draw new stuff
                pm1 = timer()

                if show_animation == True:
                    row_indices, col_indices = np.nonzero(map != 1)
                    row_boundaries, col_boundaries = np.nonzero(zero_tree)
                    plt.plot(col_indices, row_indices, "xc")
                    plt.plot(col_boundaries, row_boundaries, ".r")
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                lambda event: [exit(
                                                    0) if event.key == 'escape' else None])
                    plt.pause(5)

                pm2 = timer()

                temp_animation_time += (pm2 - pm1)

                return [laplace_time, edge_detection_time, gradient_descent_time, drawing_time, exploring_iterations]


        m2 = timer()

        laplace_time += (m2 - m1)
        laplace_time -= temp_animation_time


        # Edge Detection
        m1 = timer()

        new_map = map.copy()
        new_map[new_map != 1] = 200
        new_map[new_map == 1] = 0 
        new_map = np.uint8(new_map)
       
        # Go from PIL to CV2
        edge = cv2.Canny(new_map, 240, 250) 
        
        # Go from CV2 to PIL
        edge = Image.fromarray(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))

        # Just for drawing the Edge Detection stuff
        temp_time = 0
        
        if no_draw == False:
            tm1 = timer()

            drawing_edge = edge.copy()

            bh, bw = extra_boundary.shape
            for by in range(bh):
                for bx in range(bw):
                    draw_check = False
                    
                    if drawing_edge.getpixel((bx, by)) == (255, 255, 255):
                        draw_check = True

                    extra_check = False
                    if(extra_boundary[by][bx] >= 1):
                        drawing_edge.putpixel((bx, by), (0, 127, 0))
                        extra_check = True

                    if(boundary[by][bx] == 1):
                        drawing_edge.putpixel((bx, by), (0, 0, 127))

                    if draw_check == True:
                        if extra_check == True:
                            drawing_edge.putpixel((bx, by), (127, 0, 0))
                        else:
                            drawing_edge.putpixel((bx, by), (255, 255, 255))
                        
            result_edges.append(drawing_edge)

            tm2 = timer()

            temp_time += (tm2 - tm1)


        # Convert to grayscale image
        new_edge = ImageOps.grayscale(edge)

        # Get array of edges
        edgeArray = np.asarray(new_edge)
        
        # Find edges that aren't near the image boundaries/edges or the black obstacles
        edges = np.argwhere(edgeArray - extra_boundary == 255)

        m2 = timer()

        edge_detection_time += (m2 - m1)
        edge_detection_time -= temp_time
        drawing_time += temp_time

        # Checking to see if the entire map has been explored
        if len(edges) == 0 and map_explored == False:
            print("It took " + str(total_iter) + " RRT iterations to explore the whole map!")
            exploring_iterations = total_iter
            map_explored = True
            # return [laplace_time, edge_detection_time, gradient_descent_time, drawing_time, exploring_iterations]


        # More drawing, I think this draws the red nodes and stuff, not the Edge Detection stuff
        if no_draw == False:
            m1 = timer()

            test_im3 = Image.open(file_name)
            w3, h3 = test_im3.size

            test_im3 = test_im3.resize((round(w3 / scaleDownFactor), round(h3 / scaleDownFactor))) # just to make the image smaller
            testResult3 = test_im3.copy()

            tw3, th3 = testResult3.size
            # i have to iterate over every pixel in the image for this
            # Is there a faster way than this?
            for ti in range(tw3):
                for tj in range(th3):
                    tgs3 = 0 # black
                    if(map[tj][ti] != 1): # NumPy array of potential
                        tgs3 = max(255 - round(map[tj][ti] * 255), 50)

                    # YOU CAN CREATE PIL IMAGE FROM NUMPY ARRAY!!!
                    testResult3.putpixel((ti, tj), (tgs3, tgs3, tgs3)) # PIL Image


            videoResult = testResult3.copy()
            bh, bw = boundary.shape
            for by in range(bh):
                for bx in range(bw):
                    if(boundary[by][bx] == 1):
                        videoResult.putpixel((bx, by), (0, 0, 127))
        
            draw_result(image, node_list, videoResult, start, end)

            m2 = timer()

            drawing_time += (m2 - m1)



        # Gradient Descent
        gm1 = timer()

        temp_drawing_time = 0

        for _ in range(bpl):
            # Keep track of how many RRT iterations there have been
            total_iter = total_iter + 1

            if(total_iter >= RRTIterations):
                print("Iteration limit exceeded.")
                return [laplace_time, edge_detection_time, gradient_descent_time, drawing_time, exploring_iterations]
            

            # Get a random point!
            new_x, new_y = random_point(image, height, length, edges, extra_boundary)

            tree_x = 0
            tree_y = 0

            poser = new_y
            posec = new_x

            limit = 0
            new_node_list = []
            check2 = False # check to see if there was a gradient found between the random point and zero tree

            while True:
                # this is just for stopping the program if there's some bug in Gradient Descent
                if limit > 800:
                    break

                # Taking a step in Gradient Descent
                gradr = map[round(poser+1)][round(posec)] - map[round(poser-1)][round(posec)]
                gradc = map[round(poser)][round(posec+1)] - map[round(poser)][round(posec-1)]
                maggrad = math.sqrt(gradr**2 + gradc**2)

                if(maggrad != 0):
                    a = step_size/maggrad # scale to pixel

                    poser = poser-a*gradr
                    posec = posec-a*gradc


                # If there is no gradient/no motion, throw away this chosen random point!
                if(poser != new_y and posec != new_x):
                    # Creating the new red node
                    new_node_list.append(Node(posec, poser))
                    zero_tree[round(poser)][round(posec)] = 1


                    # temp2_drawing_time = 0

                    # # for matplotlib!!!
                    # if show_animation:
                    #     pm1 = timer()

                    #     plt.plot(round(posec), round(poser), "xc")
                    #     plt.pause(0.00001)

                    #     pm2 = timer()

                    # temp2_drawing_time += (pm2 - pm1)


                    check2 = True


                    # Drawing stuff for Gradient Descent
                    if no_draw == False:
                        dm1 = timer()

                        # Put image in video
                        vResult = videoResult.copy()
                        
                        draw_result(image, new_node_list, vResult, start, end)
                        for x in range(round(posec) - 2, round(posec) + 2):
                            for y in range(round(poser) - 2, round(poser) + 2):
                                if(0 < x and x < length - 1 and 0 < y and y < height - 1):
                                    vResult.putpixel((x, y), (255, 0, 0))

                        vResult = vResult.resize((w, h)) # make image large again
                        result_images.append(vResult)

                        dm2 = timer()
                        temp_drawing_time += (dm2 - dm1)
                
                # # Stop when end point is covered by gray area
                # if map[end[1]][end[0]] != 1:
                #     print("End point covered!")
                #     return [laplace_time, edge_detection_time, gradient_descent_time, drawing_time, exploring_iterations]

                # Stopping Gradient Descent once it has reached the Zero Tree (all the red nodes)
                break_check = False
                for sy in range(0, 2):
                    for sx in range(0, 2):
                        if map[math.floor(poser)+sy][math.floor(posec)+sx] == 0:
                            break_check = True
                if break_check == True:
                    break

                limit = limit + 1

            
            # Updating the node list and zero tree (red points)
            if check2 == True:
                pt = Node(new_x, new_y)
                pt.point = True
                node_list.append(pt)
                zero_tree[round(new_y)][round(new_x)] = 1

                middle_time = timer()
                times.append(middle_time - start_time)
                dist, _ = dist_and_angle(tree_x, tree_y, new_x, new_y)
                distances.append(dist)
            else:
                middle_time = timer()
                times.append(middle_time - start_time)
                distances.append(0)

            node_list.extend(new_node_list)


        gm2 = timer()

        gradient_descent_time += (gm2 - gm1)
        gradient_descent_time -= temp_drawing_time
        # gradient_descent_time -= temp2_drawing_time
        drawing_time += temp_drawing_time
        # drawing_time += temp2_drawing_time

        

# Drawing Function for drawing all the nodes
def draw_result(image, node_list, result, start, end):
    height = len(image)
    length = len(image[0])

    # Draw all of the nodes in the RRT
    # "Anti-aliasing"? when drawing pixel box, maybe not fill in 100% or brightness 100%?
    for node in node_list:
        round_x = round(node.x)
        round_y = round(node.y)

        if node.point == False:
            if(0 < round_x and round_x < length - 1 and 0 < round_y and round_y < height - 1):
                result.putpixel((round_x, round_y), (255, 0, 0))
        else:
            # for x in range(round_x - 2, round_x + 3):
            #     for y in range(round_y - 2, round_y + 3):
            #         if(0 < x and x < length - 1 and 0 < y and y < height - 1):
            #             result.putpixel((x, y), (255, 0, 0))
            
            if(0 < round_x and round_x < length - 1 and 0 < round_y and round_y < height - 1):
                result.putpixel((round_x, round_y), (255, 0, 0))

    # Draw the start point
    # for x in range(start[0] - 1, start[0] + 1):
    #     for y in range(start[1] - 1, start[1] + 1):
    #         if(0 < x and x < length - 1 and 0 < y and y < height - 1):
    #             result.putpixel((x, y), (0, 0, 255))
    if(0 < start[0] and start[0] < length - 1 and 0 < start[1] and start[1] < height - 1):
        result.putpixel((start[0], start[1]), (0, 0, 255))

    # Draw the end point
    if(0 < end[0] and end[0] < length - 1 and 0 < end[1] and end[1] < height - 1):
        result.putpixel((end[0], end[1]), (0, 255, 0))


# RRT-Laplace Algorithm Setup
def RRTLaplaceFunction(image, scaleDownFactor, start, end, RRTIterations, laplaceIterations, step_size, output_folder, output_path, 
               fps, output_image, output_plot, data_file, parameter_file, time_file, time_plot_file, output_edge, bpl, no_draw, show_animation):
    
    # Result Images for Video:
    result_images = []
    result_edges = []

    # Stuff for Graphs/Plots
    times = []
    distances = []

    
    file_name = image

    # Get image and convert to 2D array
    im = Image.open(image)
    w, h = im.size

    # Resizing to make algorithm run faster
    im = im.resize((round(w / scaleDownFactor), round(h / scaleDownFactor))) # just to make the image smaller

    image = np.asarray(im)

    height = len(image)
    length = len(image[0])


    start_comma = start.index(',')
    end_comma = end.index(',')

    start_first_number = int(start[1:start_comma])
    start_second_number = int(start[start_comma+2:len(start)-1])
    end_first_number = int(end[1:end_comma])
    end_second_number = int(end[end_comma+2:len(end)-1])

    start = (start_first_number, start_second_number)
    end = (end_first_number, end_second_number)


    if(start_first_number < 0 or start_first_number >= length or start_second_number < 0 or start_second_number >= height):
        raise Exception("Sorry, the inputted end point is not in the image!")


    startTime = timer()

    # Node List!
    node_list = [] 
    node_list.append(Node(start[0], start[1]))

    # Map of all the potential in the image through the algorithm (i think)
    potential_map = np.ones((height, length)) 

    # Get locations of where the black walls/pixels are!
    boundary = np.zeros((height, length))

    for y in range(height):
        for x in range(length):
            
            if(image[y][x][0] <= 10):
                if(start_first_number == x and start_second_number == y):
                    raise Exception("Sorry, the inputted end point is on the boundary!")

                boundary[y][x] = 1


    # for matplotlib
    sx = 30.0  # [m]
    sy = 30.0  # [m]
    gx = 70.0  # [m]
    gy = 70.0  # [m]

    ox, oy = [], []
    for i in range(10, 80):
        ox.append(i)
        oy.append(10.0)
    for i in range(10, 80):
        ox.append(80.0)
        oy.append(i)
    for i in range(10, 81):
        ox.append(i)
        oy.append(80.0)
    for i in range(10, 81):
        ox.append(10.0)
        oy.append(i)
    for i in range(10, 60):
        ox.append(40.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(60.0)
        oy.append(80.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        # plt.show()


    # Computing the extra boundary (green area in video) to make sure Edge Detection works properly
    edge_kernel = np.array([[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]])
    extra_boundary = []
    extra_boundary = cv2.filter2D(boundary, -1, edge_kernel)
    extra_boundary[extra_boundary >= 1] = 255
    extra_boundary[:, [0, 1, -1, -2]] = 255
    extra_boundary[[0, 1, -1, -2], :] = 255

  
    # Run the RRT-Laplace Algorithm
    lt, edt, gdt, dt, ei = RRT(image, node_list, potential_map, boundary, extra_boundary, start, end, RRTIterations, laplaceIterations, step_size, 
                       file_name, scaleDownFactor, startTime, result_images, times, distances, bpl, no_draw, w, h, result_edges, show_animation)
    

    endTime = timer()

    # Print all the time components!
    # print(lt, edt, gdt, dt)


    # Drawing the final result image!
    result = im.copy() 
    draw_result(image, node_list, result, start, end)
    result = result.resize((w, h)) # make image go back to its original dimensions


    # More drawing/video creation stuff!
    if no_draw == False:
        result_images.append(result)

        # Video for the red points, showing gradient descent
        writer = imageio.get_writer(output_folder + "/" + output_path, fps=fps) 

        for image_filename in result_images:
            writer.append_data(np.array(image_filename))

        writer.close()

        # Video for showing how the Edge Detection works
        writer = imageio.get_writer(output_folder + "/" + output_edge, fps=2) 

        for image_filename in result_edges:
            writer.append_data(np.array(image_filename))

        writer.close()
    

    # ALL THE REST BELOW IS JUST CREATING THE PLOTS AND SAVING EVERYTHING INTO FILES
    # ALL THE REST BELOW IS JUST CREATING THE PLOTS AND SAVING EVERYTHING INTO FILES
    # ALL THE REST BELOW IS JUST CREATING THE PLOTS AND SAVING EVERYTHING INTO FILES

    # Show and save the tree image
    result.save(output_folder + "/" + output_image)

    # Generate plots and save the data to .csv file
    total_distances = list(accumulate(distances))

    x = np.array(times)
    y = np.array(distances)
    total_y = np.array(total_distances)

    data = [x, y, total_y]
    np.savetxt(output_folder + "/" + data_file, data, delimiter = ",")

    # # Gradient Distances vs. Time
    # plt.scatter(x, y)
    # plt.title("Gradient Distances vs. Time")
    # plt.xlabel("Time (in seconds)")
    # plt.ylabel("Gradient Distances (in pixels)")

    # a = np.polyfit(x, y, 1)
    # b = np.poly1d(a)

    # plt.plot(x, b(x))

    # plt.savefig(output_folder + "/" + "gdt" + output_plot)

    # plt.close()


    # # Total Gradient Distances vs. Time
    # plt.scatter(x, total_y)
    # plt.title("Total Gradient Distances vs. Time")
    # plt.xlabel("Time (in seconds)")
    # plt.ylabel("Total Gradient Distances (in pixels)")

    # c = np.polyfit(x, total_y, 1)
    # d = np.poly1d(c)

    # plt.plot(x, d(x))

    # plt.savefig(output_folder + "/" + "tgdt" + output_plot)

    # plt.close()

    
    # # Histogram
    # plt.hist(y, bins = 30, edgecolor='black')
    # plt.title("Histogram of Gradient Distances (in pixels)")
    # plt.xlabel('Gradient Distances')
    # plt.ylabel('Count of Gradient Distances')

    
    # plt.savefig(output_folder + "/" + "hist" + output_plot)

    # plt.close()



    # Output parameters to txt file
    file = open(output_folder + "/" + parameter_file, "w")
    file.write("Enter the name of your image file (include .jpg, .png, etc.): " + file_name + "\n")
    file.write("Dimensions of image: " + str(w) + " pixels by " + str(h) + " pixels!" + "\n")
    file.write("Input the factor for scaling down the image (4 recommended): " + str(scaleDownFactor) + "\n")
    file.write("Dimensions of scaled image: " + str(round(w / scaleDownFactor)) + " pixels by " + str(round(h / scaleDownFactor)) + " pixels!" + "\n")
    file.write('Enter your start point (unit: pixel) in the form of "(x, y)": (' + str(start_first_number) + ', ' + str(start_second_number) + ')' + '\n')
    file.write('Enter the number of RRT iterations (10000 recommended): ' + str(RRTIterations) + '\n')
    file.write('Enter the number of Laplace equation iterations (250 recommended): ' + str(laplaceIterations) + '\n')
    file.write('Enter the step size of the RRT (3 recommended): ' + str(step_size) + "\n")
    file.write('Enter the name of the output folder: ' + output_folder + '\n')
    file.write('Enter the desired video file name (include .mp4 at the end, make it descriptive based on the parameters entered): ' + output_path + '\n')
    file.write('Enter the fps of video (recommended 10 for small videos, 30+ for longer): ' + str(fps) + '\n')
    file.write('Enter the output tree image file (include .png at the end): ' + output_image + '\n')
    file.write('Enter the output plot image file (include .png at the end): ' + output_plot + '\n')
    file.write('Enter the desired data file (include .csv at the end): ' + data_file + '\n')
    file.write('Iteration limit exceeded.' + '\n')
    file.write('Total Time: This took ' + str(round(endTime-startTime, 3)) + ' seconds!')
    file.close()

    # Output times to txt file
    file = open(output_folder + "/" + time_file, "w")
    file.write("Laplace Time: " + str(lt) + "\n")
    file.write("Edge Detection Time: " + str(edt) + "\n")
    file.write("Gradient Descent Time: " + str(gdt) + "\n")
    file.write("Drawing Time: " + str(dt) + "\n")
    file.write("Total Time: " + str(endTime-startTime) + "\n")
    file.write("It took " + str(ei) + " RRT Iterations to explore the whole map!" + "\n")
    file.close()

    # Plot times 
    # plt.plot([file_name], [endTime-startTime], marker='o', linestyle='None', label="Total Time")
    # plt.plot([file_name], [lt], marker='o', linestyle='None', label="Laplace Time")
    # plt.plot([file_name], [edt], marker='o', linestyle='None', label="Edge Detection Time")
    # plt.plot([file_name], [gdt], marker='o', linestyle='None', label="Gradient Descent Time")
    # plt.plot([file_name], [dt], marker='o', linestyle='None', label="Drawing Time")
    # plt.legend()
    # plt.savefig(output_folder + "/" + time_plot_file)
    # plt.close()

    # Return data to create a combined .csv file
    return (RRTIterations, laplaceIterations, file_name, scaleDownFactor, bpl, start, endTime-startTime, lt, edt, gdt, dt, ei, times, distances, total_distances)
    

# Main Function, specify the parameters you want to test!
def main():
    conda = input("This is just for VSCode w/ Conda Python version, type anything here to start: ")

    images = ['metrics'] # What image do you want to use?
    RRTIterations = [int(500)] # How many random points do you want?
    laplaceIterations = [int(20)] # How many times do you want to run the Laplace Equation per random point?

    scaleDownFactor = [1] # By how much do you want to scale down the image?
    start = ['(30, 30)'] # Where do you want the start point to be?
    end = ['(70, 70)'] # Where do you want the end point to be?

    step_size = int(1) # Don't change this.
    output_folder = 'may14_videos' # In which folder do you want to save the files?

    fps = int(120) # What FPS do you want your Gradient Descent video to be in?

    bpl = [int(5)] # How many random points per batch of Laplace Equation runs do you want to have?

    no_draw = True # False = You get videos, True = You don't get videos
    
    show_animation = True # do you want the matplotlib animation to be shown?



    number = 1
    for i in range(len(images)):
        for j in range(len(RRTIterations)):
            for k in range(len(laplaceIterations)):
                for l in range(len(bpl)):
                    for m in range(len(scaleDownFactor)): # also for start point
                        specific = '_ye_' + images[i] + '_rrt' + str(RRTIterations[j]) + '_lap' + str(laplaceIterations[k]) + '_bql' + str(bpl[l]) + '_sdf' + str(scaleDownFactor[m])

                        output_path = 'vid' + specific + '.mp4'
                        output_edge = 'edge' + specific + '.mp4'
                        output_image = 'res' + specific + '.png'
                        output_plot = 'plot' + specific + '.png'
                        data_file = 'data' + specific + '.csv'
                        parameter_file = 'param' + specific + '.txt'
                        time_file = 'time' + specific + '.txt'
                        time_plot_file = 'timeplot' + specific + '.png'


                        rrt, lp, file_name, sdf, bplr, startr, tt, lt, edt, gdt, dt, ei, times, distances, total_distances = RRTLaplaceFunction(images[i] + '.png', scaleDownFactor[m], start[m], end[m], RRTIterations[j], laplaceIterations[k], 
                                        step_size, output_folder, output_path, fps, output_image, output_plot, 
                                        data_file, parameter_file, time_file, time_plot_file, output_edge, bpl[l], no_draw, show_animation)
                        
                        print('World Used: ' + file_name) 
                        print('Laplace Iterations: ' + str(lp))
                        print('Scale Down Factor: ' + str(sdf))
                        print('Branches per Laplace: ' + str(bplr))
                        
                        print('Total Time: ' + str(tt))
                        print('Laplace Time: ' + str(lt))
                        print('Edge Detection Time: ' + str(edt))
                        print('Gradient Descent Time: ' + str(gdt))
                        print('Drawing Time: ' + str(dt))
                        print('It took ' + str(ei) + ' random points/iterations to explore the entire world!')

                        
                        number = number + 1





if __name__ == '__main__':
    main()


