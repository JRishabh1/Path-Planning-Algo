import pygame
from RRTDef import RRTMap
from RRTDef import RRTGraph
import time


def main():

    dims = (600, 1000)
    start = (50, 50)
    goal = (510, 510)
    obsDim = 30
    obsNum = 50
    iter  = 0
    initTime = 0

    pygame.init()
    map = RRTMap(start, goal, dims, obsDim, obsNum)
    graph = RRTGraph(start, goal, dims, obsDim, obsNum)

    obstacles = graph.addobs()

    map.drawMap(obstacles)

    initTime = time.time()
    while (not graph.pathToGoal()):
        elapsedTime = time.time() - initTime
        initTime = time.time()
        if elapsedTime > 10:
            raise
        if iter % 10 == 0:
            X, Y, Parent = graph.bias(goal)
            pygame.draw.circle(map.map, map.red, (X[-1], Y[-1]), map.nodeRadius+1, 0)
            pygame.draw.line(map.map, map.blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        else: 
            X, Y, Parent = graph.expand()
            pygame.draw.circle(map.map, map.red, (X[-1], Y[-1]), map.nodeRadius+1, 0)
            pygame.draw.line(map.map, map.blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]), map.edgeThickness)
        if iter % 5 == 0:
            pygame.display.update()
        iter += 1
    map.drawPath(graph.getPathCoords())
    pygame.display.update()
    pygame.event.clear()
    pygame.event.wait(0)
    

if __name__ == '__main__':
    result = False
    while not result:
        try:
            main()
            result = True
        except:
            result = False
    