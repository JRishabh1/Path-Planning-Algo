#Base class on which all RRT algos are based

import math
import random
import pygame

#Draws Map with Obstacles and Path Generated by RRT

class RRTMap:

    def __init__(self, start, goal, mapDim, obsDim, obsNum):
        self.start = start
        self.goal = goal
        self.mapHeight, self.mapWidth = mapDim
        
        self.obstacles = []
        self.obsDim = obsDim
        self.obsNum = obsNum

        #Colors

        self.red = (255, 0, 0)
        self.blue = (0, 255, 0)
        self.green = (0, 0, 255)
        self.white = (255, 255, 255)        
        self.black = (0, 0, 0)
        self.gray = (70, 70, 70)

        #Setting Window

        self.mapWindowName = 'RRT'
        pygame.display.set_caption(self.mapWindowName)
        self.map = pygame.display.set_mode((self.mapWidth,self.mapHeight))
        self.map.fill((255, 255, 255))
        self.nodeRadius = 3
        self.nodeThickness = 0
        self.edgeThickness = 1

    #Draws start and goal as circles

    def drawMap(self, obstacles):
        pygame.draw.circle(self.map, self.green, self.start, self.nodeRadius+5, 0)
        pygame.draw.circle(self.map, self.green, self.goal, self.nodeRadius+20, 1)
        self.drawobs(obstacles)

    def drawobs(self, obstacles):
        obsList = obstacles.copy()
        while (len(obsList) > 0):
            obstacle = obsList.pop(0)
            pygame.draw.rect(self.map, self.gray, obstacle)

    def drawPath(self, path):
        for node in path:
            pygame.draw.circle(self.map, self.red, node, self.nodeRadius+2, 0)

#Does an assortment of tasks from constructing the obstacles and path to finding nearest neighbours 

class RRTGraph:
    
    def __init__(self, start, goal, mapDim, obsDim, obsNum):
        (x,y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.mapHeight, self.mapWidth = mapDim
        self.x = []
        self.y = []
        self.parent = []

        #Initialize Tree

        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)
                
        #Initialize Obstacles

        self.obs = []
        self.obsDim = obsDim
        self.obsNum = obsNum

        #Initialize Path data

        self.goalState = None
        self.path = []


    # n: Instance of node
    # x: X coordinate
    # y: Y coordinate
        
    def addNode(n, x, y):
        self.x.insert(x)
        self.y.append(y)

    def deleteNode(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def nodeCount(self):
        return len(self.x)

    def addEdge(self, parent, child):
        self.parent.insert(child, parent)

    def deleteEdge(self, n):
        self.parent.pop(n)


    def addobs(self):
        obs = []

        for i in range(0, self.obsNum):
            #temporarily hold obstacle
            rectangle = None
            #indicates whether start and goal coordinates are in obstacle
            startGoalCol = True
            while startGoalCol:
                upper = self.addRect()
                rectangle = pygame.Rect(upper, (self.obsDim, self.obsDim))
                if not rectangle.collidepoint(self.start) or not rectangle.collidepoint(self.goal):
                    self.startGoalCol = False
            obs.append(rectangle)
        self.obstacles=obs.copy()
        return obs
            
    def addRect(self):
        upperX = int(random.uniform(0, self.mapWidth - self.obsDim))
        upperY = int(random.uniform(0, self.mapHeight - self.obsDim))

        return (upperX, upperY)

    def nearestNeighbour(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear

    #Calculates Euclidean distance given 2 nodes

    def distance(self, n1, n2):
        (x1,y1) = (self.x[n1], self.y[n1])
        (x2,y2) = (self.x[n2], self.y[n2])
        return ((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)**0.5
    
    #Randomly samples the map

    def sampleEnv(self):
        x = int(random.uniform(0, self.mapWidth))
        y = int(random.uniform(0, self.mapHeight))
        return x, y

    #Checks if sampled node is in a valid coordinate or not

    def isFree(self):
        n = self.nodeCount() - 1
        (x, y) = (self.x[n], self.y[n])
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectangle = obs.pop(0)
            if rectangle.collidepoint(x,y):
                self.deleteNode(n)
                return False
        return True

#Uses interpolation to generate checkpoints to check if a proposed path collides with a path

    def crossobs(self, x1, x2, y1, y2):
        obs = self.obstacles.copy()
        while len(obs) > 0:
            rectangle = obs.pop(0)
            for i in range (0, 101):
                u = i/100
                x = x1*u + x2 * (1-u)
                y = y1 * u + y2 * (1-u)
                if rectangle.collidepoint(x,y):
                    return True
        return False

    def connect(self, n1, n2):
        (x1,y1) = (self.x[n1], self.y[n1])
        (x2,y2) = (self.x[n2], self.y[n2])
        if self.crossobs(x1, x2, y1, y2):
            self.removeNode(n2)
            return False
        self.addEdge(n1, n2)
        return True

    #Creates new node in a max radius around old node

    def step(self, nnear, nrand, dmax = 35):
        dist = self.distance(nnear, nrand)
        if dist > dmax:
            u = dmax/dist
            (xnear, ynear) = (self.x[near], self.y[near])
            (xrand, yrand) = (self.x[nrand], self.y[nrand])
            (px, py) = (xrand - xnear, yrand - ynear)
            theta = math.atan2(py, px)
            (x, y) = (int(xnear+ dmax * math.cos(theta)), int(ynear+ dmax * math.sin(theta)))
            self.deleteNode(nrand)
            if abs(x - self.goal[0]) < dmax and abs(y - self.goal[1]) < dmax:
                self.addNode(nrand, self.goal[0], self.goal[1])
                self.goalState = nrand
                self.goalFlag = True
            else:
                self.addNode(nrand, x, y)
                        

    def pathToGoal(self):
        if self.goalFlag:
            self.path = []
            self.path.append(self.goalState)
            newPos = self.parent[self.goalState]
            while newPos != 0:
                self.path.append(newPos)
                newPos = self.parent[self.newPos]
            self.path.append(0)
        return self.goalFlag


    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.append((x, y))
        return pathCoords

    def bias(self, goal):
        n = self.nodeCount()
        self.addNode(n, goal[0], goal[1])
        nnear = self.nearestNeighbour(n)
        self.step(nnear, n)
        self.connect(nnear, n)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.nodeCount()
        x, y = self.sampleEnv()
        self.addNode(n, x, y)
        if self.isFree():
            xNearest = self.nearestNeighbour(n)
            self.step(xNearest, n)
            self.connect(xNearest, n)
        return self.x, self.y, self.parent
            
    def cost(self):
        pass

