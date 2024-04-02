from numpy import *
from random import random,seed
from PIL import Image, ImageDraw
from time import time
from numba import jit  , prange 
import cv2
from tqdm import tqdm 
seed(323)

G = 1.0 # Gravitational Constant
THRESHOLD = 1 # Number of bodies after which to subdivide Quad
MAXDEPTH = 10
THETA = 0.5 # Barnes-Hut ratio of accuracy
ETA = 0.5 # Softening factor

NUM_CHECKS = 0 # Counter

class QuadTree:
    """ Class container for for N points stored as a 2D Quadtree """
    root = None
    def __init__(self, bbox, N, theta = THETA):
        self.bbox = bbox
        self.N = N
        self.theta = theta
        self.reset()
    def reset(self):
        self.root = Quad(self.bbox)
    def generate(self):
        # Build up nodes of tree fresh for bodies
        self.reset()
        for x in range(self.N):
            # For each body, add to root
            self.root.addBody(x,0)

    def updateSys(self, dt):
        self.calculateBodyAccels()
        global VEL,POS
        VEL += ACC * dt
        POS += VEL * dt
    
    def calculateBodyAccels(self):
        # Update ACC table based on current POS
        for k in range(self.N):
            ACC[k] = self.calculateBodyAccel(k)
    
    def calculateBodyAccel(self, bodI):
        return self.calculateBodyAccelR(bodI, self.root)

    def calculateBodyAccelR(self, bodI, node):
        # Calculate acceleration on body I
        # key difference is that body is ignored in calculations
        acc = zeros(2,dtype=float)
        if (node.leaf):
            for k in node.bods:
                if k != bodI: # Skip same body
                    acc += getForce(POS[bodI], 1.0, POS[k], MASS[k])
        else:
            s = max(node.bbox.sideLength)
            d = node.center - POS[bodI]
            r = sqrt(d.dot(d))
            if (r > 0 and s/r < self.theta):
                acc += getForce(POS[bodI], 1.0, node.com, node.mass)
            else:
                for k in range(4):
                    if node.children[k] != None:
                        acc += self.calculateBodyAccelR(bodI, node.children[k])
        return acc

    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return self.root.__str__()
        

# def getForce(p1,m1,p2,m2):
#     d = p2 - p1
#     r = sqrt(d.dot(d)) + ETA
#     f = array(d * G * m1 * m2 / r**3)
#     global NUM_CHECKS
#     NUM_CHECKS += 1
#     return f

@jit(nopython=True,parallel=True)  
def getForce(p1, m1, p2, m2):
    d = p2 - p1
    r = sqrt(d.dot(d)) + ETA
    f = empty(2, dtype=float64)
    f[0] = d[0] * G * m1 * m2 / r**3
    f[1] = d[1] * G * m1 * m2 / r**3
    return f

class Quad:
    """ A rectangle of space, contains point bodies """
    def __init__(self,bbox,bod = None,depth=0):
        self.bbox = bbox
        self.center = bbox.center
        self.leaf = True # Whether is a parent or not
        self.depth = depth
        if bod != None:
            self.setToBody(bod)
            self.N = 1
        else:
            self.bods = []
            self.mass = 0.
            self.com = array([0,0], dtype=float)
            self.N = 0
            
        self.children = [None]*4 # top-left,top-right,bot-left,bot-right

    def addBody(self, idx,depth):
        if len(self.bods) > 0 or not self.leaf:
            if (depth >= MAXDEPTH):
                self.bods.append(idx)
            else:
                subBods = [idx]
                if len(self.bods) > 0:
                    subBods.append(self.bods[0])
                    self.bods = []

                for bod in subBods:
                    quadIdx = self.getQuadIndex(bod)
                    if self.children[quadIdx]:
                        self.children[quadIdx].addBody(bod,depth+1)
                    else:
                        subBBox = self.bbox.getSubQuad(quadIdx)
                        self.children[quadIdx] = Quad(subBBox, bod,depth+1)

                self.leaf = False

            bodyMass = MASS[idx]
            self.com = (self.com * self.mass + POS[idx] * bodyMass) / (self.mass + bodyMass)
            self.mass += bodyMass
        else:
            self.setToBody(idx)

        self.N += 1
    
    def setToBody(self,idx):
        self.bods = [idx]
        self.mass = float(MASS[idx].copy())
        self.com  = POS[idx].copy()

    def getQuadIndex(self,idx):
        return self.bbox.getQuadIdx(POS[idx])
        
    def __repr__(self):
        return self.__str__()
    def __str__(self):        
        if len(self.bods) > 0:
            bodstring = str(self.bods)
        else:
            bodstring = "PARENT"
        if any(self.children):
            childCount = "C:%g," % sum(map(lambda x: 1 if x else 0, self.children))
            childStr = "\n"
            for x in range(4):
                childStr += ("-"*(self.depth+1))+str(x+1)+" "
                if self.children[x]:
                    childStr += str(self.children[x])
                if x < 3: 
                    childStr += "\n"

        else:
            childCount = ""
            childStr = ""
        return "D%g{N:%g,M:%g,%sCOM:%s,B:%s}%s" % (self.depth,self.N,round(self.mass,2),childCount,self.com.round(2),bodstring,childStr)


class BoundingBox:
    def __init__(self,box,dim=2):
        assert(dim*2 == len(box))
        self.box = array(box,dtype=float)
        self.center = array([(self.box[2]+self.box[0])/2, (self.box[3]+self.box[1])/2], dtype=float)
        self.dim = dim
        self.sideLength = self.max() - self.min()

    def max(self):
        return self.box[self.dim:]
    def min(self):
        return self.box[:self.dim]
    def inside(self,p):
        if any(p < self.min()) or any(p > self.max()):
            return False
        else:
            return True
    def getQuadIdx(self,p):
        if p[0] > self.center[0]: # x > mid
            if p[1] > self.center[1]: # y > mid
                return 1
            else:
                return 3
        else:
            if p[1] > self.center[1]: # y > mid
                return 0
            else:
                return 2
    def getSubQuad(self,idx):
        b = array([None,None,None,None])
        if idx % 2 == 0:
            b[::2] = [self.box[0], self.center[0]] # x - midx
        else:
            b[::2] = [self.center[0], self.box[2]] # midx - x2
        if idx < 2:
            b[1::2] = [self.center[1], self.box[3]] # midy - y2
        else:
            b[1::2] = [self.box[1], self.center[1]] # y - midy
        return BoundingBox(b,self.dim)
def convertPos(p):
    # From BOUNDS to IMAGE_SIZE and list format [x,y]
    # p = (x,y) -> [ix, iy]
    c = trunc( ( p - BOUNDS.min()) / (BOUNDS.max()-BOUNDS.min()) * array(VIDEO_FRAME_SIZE) ).tolist()
    c[1] = VIDEO_FRAME_SIZE[1] - c[1] # flip y axis
    return c

def drawBodies(drawPtr):
    for x in range(N):
        if BOUNDS.inside(POS[x]):
            p = convertPos(POS[x])
            r = MASS[x] * 5
            drawPtr.ellipse(join(p-r,p+r), fill=(0,0,0))

def join(p1,p2):
    return ([p1[0],p1[1],p2[0],p2[1]])

##############################################
##############################################

N = 1000
BOUNDS = BoundingBox([0,0,20,20])

# Global variables
# 2D Position
MASS = zeros(N,dtype=float)
POS = zeros((N,2),dtype=float)
VEL = zeros((N,2),dtype=float)
ACC = zeros((N,2),dtype=float)
for i in range(N):
    MASS[i] = 1
    POS[i] = BOUNDS.min() + array([random(),random()])*BOUNDS.sideLength

sys = QuadTree(BOUNDS, N)

# Simulation parameters
DT = 0.1
NUM_FRAMES = 1000

# Video parameters
VIDEO_FILENAME = "gravitational_simulation.mp4"
VIDEO_FPS = 60
VIDEO_FRAME_SIZE = (1920, 1080)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, VIDEO_FPS, VIDEO_FRAME_SIZE)

# Run simulation and save frames
for _ in tqdm(range(NUM_FRAMES)):
    # Generate quadtree and update system
    sys.generate()
    sys.updateSys(DT)

    # Create image frame
    frame = Image.new('RGB', VIDEO_FRAME_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(frame)
    drawBodies(draw)

    # Convert image to OpenCV format
    frame_cv2 = cv2.cvtColor(array(frame), cv2.COLOR_RGB2BGR)

    # Write frame to video
    out.write(frame_cv2)

# Release video writer
out.release()

print("Video saved as", VIDEO_FILENAME)
