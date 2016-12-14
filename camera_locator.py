#!/usr/bin/env python
"""
SYNOPSIS

    TODO helloworld [-h,--help] [-v,--verbose] [--version]

DESCRIPTION

    TODO This describes how to use this script. This docstring
    will be printed by the script if there is an error or
    if the user requests help (-h or --help).

EXAMPLES

    to run all jpgs in imgs directory:

        python camera_locator.py

    to run one jpg

        python camera_locator.py -f /path/to/image.jpg


AUTHOR
    joeyparker47@gmail.com
"""

import sys, os, traceback, optparse, time, glob

from scipy.spatial.distance import pdist, cdist, squareform

from matplotlib import rcParams
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure


if sys.version_info[0] < 3:
    import Tkinter as Tk
    from Tkinter import messagebox
else:
    import tkinter as Tk
    from tkinter import messagebox

from PIL import Image, ImageTk


def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return the intersection
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


def has3Children(hierarchy,h,count=1):
    '''
    return true if contour has exactly 3 children
    how contour's hierarchys are represented is explained here: http://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html    
    '''

    if count > 3:
        return False

    next = h[2]
    if next == -1:
        if count == 3:
            return True
        return False

    return has3Children(hierarchy,hierarchy[next],count+1)


def order_points(pts):
    '''
    Returns the points in a specific way, so the world_array in getCameraLocationInfo will be lined up properly
    param pts = [A,C,B,N]

    return pts should be [A*, C, B*, N]
    where A* is whichever of A and B is counter clockwise of C.
    B* is whichever is clockwise of C.
         |C  |     |  B|
         |_ _|     |___|
         |             |
         |___          |
         |   |         |
         |A__|________N|
    '''

    A = pts[0]
    B = pts[2]

#   if C[y]      < N[y]
    if pts[1][1] < pts[3][1]:
        
        if A[0] > B[0]:
            return pts
        return np.array([B, pts[1], A, pts[3]], dtype="float32")

    else:

        if A[0] > B[0]:
            return np.array([B, pts[1], A, pts[3]], dtype="float32")

        return pts

def getCameraLocationInfo(corners,sizes):

    w = 4.4 #half width of the QR code, in cm
    # corners of the QR code in its coordinates
    # in the order: A C B N
    world_array = np.array([
        [-w,w,0],
        [-w,-w,0],
        [w,-w,0],
        [w,w,0]
        ], dtype=np.float64)


    camera_positions = []
    camera_pointsAt  = []
    camera_render    = []
    for i,corner_points in enumerate(corners):
    
        focal_length = sizes[i][0]
        center = (sizes[i][1]/2, sizes[i][0]/2)

        #camrea matrix, defines the nature of the camera. focal_lengthx == focal_lengthy is assumed, meaning pixels are square
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )


        #points of the QR code in the image
        image_points  = order_points(corner_points)
        cameraMatrix  = np.eye(3) 
        ret,rvec,tvec = cv2.solvePnP(world_array,image_points,camera_matrix,np.array([]))
        rotM_cam = cv2.Rodrigues(rvec)[0]
        
        #rvec,tvec, the rotational matrix and translational array describing going from camera to model
        #put into a 4,4 matrix and get the inverse to get rvec and tvec describing going from model to camera. 
        #camera location is then tvec
        
        b = np.zeros((4,4))
        b[:-1,:-1] = rotM_cam
        b[:-1,3]   = tvec.T
        b[3,3]     = 1
        
        b = inv(b)

        camera_position = b[:-1,3]
        camera_positions.append(camera_position)
        camera_pointsAt.append(camera_position - b[:-1,2] * (camera_position[2] / b[:-1,2][2]))

        camera_points = []

        camera_points.append(camera_position - (b[:-1,:-1].T * np.array([[2],[0.5],[0]])).sum(axis=0))
        camera_points.append(camera_position - (b[:-1,:-1].T * np.array([[-2],[0.5],[0]])).sum(axis=0))
        camera_points.append(camera_position - (b[:-1,:-1].T * np.array([[-2],[-5.5],[0]])).sum(axis=0))
        camera_points.append(camera_position - (b[:-1,:-1].T * np.array([[2],[-5.5],[0]])).sum(axis=0))
        
        camera_render.append(camera_points)

    return camera_positions, camera_pointsAt, camera_render

def getCornersOfQRCode(images):
    '''
    get the four corners of the QR code in each image in images. 
    The QR code is orginized as shown below. 

         |C  |     |  B|
         |_ _|     |___|
         |             |
         |___          |
         |   |         |
         |A__|________N|
    '''
    corners_out = []
    
    for c,im in enumerate(images):
        # get and do some initial work on the image

        print("Finding camera location, image", c)
   
 
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(imgray,200,255,0)
        
         #get contours
        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0]

        #from the contours find those which have exactly 3 children, indicating they are the QR code corners
        #corner_cooridinates are the weighted centers of those contours
        #3 corners should be found
        corner_squares = []
        for idx, h in enumerate(hierarchy):
            if has3Children(hierarchy,h):
                corner_squares.append(contours[idx])

        moments = [cv2.moments(c) for c in corner_squares]
        corners_coordinates = np.array([[m['m10']/m['m00'],m['m01']/m['m00']] for m in moments])

        #the 3 corners make approximatly a right angle triangle. Corners a and b are on the largest edge. 
        #o is a point in approximately the center of the QR
        D = pdist(corners_coordinates)
        D = squareform(D);

        #h: length of the 'hypotenuse' of the approx right angle triangle ABC
        h = D.max()/2
        summed = D.sum(axis=0)

        #c is the corner square with lowest total distance from each other corner. a and b are assigned to be the others. 
        #which is a and which is b does not matter right now, they are reoginized later in order_points()
        #these are their indices. 
        inc = summed.argmin()
        ina = (inc + 1) % 3
        inb = (ina + 1) % 3

        #o is a point in approximately the center of the QR
        #found using simple vector algebra
        c2a = corners_coordinates[ina] - corners_coordinates[inc]
        c2b = corners_coordinates[inb] - corners_coordinates[inc]
        a2b = corners_coordinates[inb] - corners_coordinates[ina]

        a2b_normed = a2b/np.linalg.norm(a2b)
        a2b = np.multiply(a2b_normed,h)
        o   = corners_coordinates[ina]+a2b


        #for each of the squares find their 4 corners by approximating a polygon. 
        corner_squares = [np.vstack(c) for c in corner_squares]
        corners_of_square = [[],[],[]]
        continue_flag = False
        for i in [ina,inb,inc]:
            poly = cv2.approxPolyDP(corner_squares[i],10,1).squeeze()
            
            #use only if poly is a square/rectangle
            if len(poly) != 4:
                continue_flag = True
                break
            corners_of_square[i] = poly
             
        if continue_flag:
            continue


        #relevant points are C, the corner of c. and the far edges of A and B, so two lines can be formed.
        #Their intersection is the remaning corner of the QR, N
        C = corners_of_square[inc][cdist(corners_of_square[inc], [o]).argmax()]

        A0 = corners_of_square[ina][cdist(corners_of_square[ina], [o]).argmax()]
        A1 = corners_of_square[ina][cdist(corners_of_square[ina], [corners_coordinates[inc] - a2b]).argmax()]
        
        B0 = corners_of_square[inb][cdist(corners_of_square[inb], [o]).argmax()]
        B1 = corners_of_square[inb][cdist(corners_of_square[inb], [corners_coordinates[inc] + a2b]).argmax()]

        #the corners are then refined for subpixel accuracy
        cornersForSubPix = np.float32(np.array([A0,A1, B0, B1]))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        cv2.cornerSubPix(imgray,cornersForSubPix,(3,3),(-1,-1),criteria)


        A  = np.array([cornersForSubPix[0],cornersForSubPix[1]])
        B  = np.array([cornersForSubPix[2],cornersForSubPix[3]])

        N = seg_intersect(A[0],A[1],B[0],B[1])

        #the 4 corners of the QR code are put into a specific order, then placed into the array to be returned
        corners_out.append(order_points(np.array([A[0], C, B[0], N], dtype="float32")) )

    return corners_out

def loadImages(filenames):

    '''
    Load images

    return cvimages for use in locating the QRcode and camera camera_position
    and tkimages, reduced in size, used for displaying
    '''
    cvimages = []
    tkimages = []
    sizes    = []
    for filename in filenames:
        print ("Loading", filename)
        im = cv2.imread(filename)
        size = im.shape
        sizes.append(size)
        cvimages.append(im)
        b,g,r  = cv2.split(im)

        original = Image.fromarray(cv2.merge((r,g,b)))
        resized = original.resize((int(size[1]/6), int(size[0]/6)),Image.ANTIALIAS)
        tkimages.append(ImageTk.PhotoImage(resized))

    return cvimages, tkimages, sizes

class tkGUI:
    '''

    GUI interface

    next and prev buttons move between images in the imgs folder. 
    save saves the current view of the plot as an image.

    '''

    def __init__(self, master, tkimages, camera_positions, camera_pointsAt, camera_render):

        self.index  = 0
        self.master = master
        self.tkimages = tkimages
        self.camera_positions = camera_positions
        self.camera_pointsAt  = camera_pointsAt
        self.camera_render    = camera_render
        

        self.drawGraph()

    def on_key_event(self,event):

   
        if event.key == 'right':
            self.next()
            return 

        if event.key == 'left':
            self.prev()
            return

    def next(self):
        self.index = (self.index + 1) % (len(self.tkimages) - 1)
        self.drawGraph()

    def prev(self):
        self.index = (self.index - 1) % (len(self.tkimages) - 1)
        self.drawGraph()

    def save(self):
        self.fig.savefig('camera_position.png.png')
        messagebox.showinfo("alert" , "Current plot saved as camera_position.png")
        print("image saved")


    def drawGraph(self):

        print("drawing graph", self.index) 

        self.fig    = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().grid(row=0,columnspan=8) #.pack(side=Tk.LEFT)  

        self.panel = Tk.Label(master=self.master, image = self.tkimages[self.index]).grid(row=0,column=8)
        #self.panel.pack(side = Tk.RIGHT)

        self.quitbutton = Tk.Button(master=self.master, text='Quit', command=self.master.quit).grid(row=1,column=0)
        self.saveButton = Tk.Button(master=self.master, text="Save Image", command=self.save).grid(row=1,column=4)

        if len(self.tkimages) > 1:
            self.prevButton = Tk.Button(master=self.master, text="Prev", command=self.prev).grid(row=1,column=3,sticky='E')
            self.nextButton = Tk.Button(master=self.master, text="Next", command=self.next).grid(row=1,column=5,sticky='W')

        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.master.protocol("WM_DELETE_WINDOW", self.master.quit)
        
        # points for QR code centered to be drawn in the plot. Centered at 0,0
        # | C |     | B |
        # |_ _|     |___|
        # |             |
        # |___          |
        # | A |         |
        # |___|________N|
        #            ---- SQUARE ------   -  C  -    -  A  -   -  B  -
        QRStart_x = [-4.4, 4.4, 4.4, 4.4,-1.4,-4.4, 4.4, 1.4,-1.4,-4.4]
        QREnd_x   = [-4.4, 4.4,-4.4,-4.4,-1.4,-1.4, 1.4, 1.4,-1.4,-1.4]
        QRStart_y = [-4.4,-4.4,-4.4, 4.4,-4.4,-1.4,-1.4,-4.4, 4.4, 1.4]
        QREnd_y   = [ 4.4, 4.4,-4.4, 4.4,-1.4,-1.4,-1.4,-1.4, 1.4, 1.4]

        #try to get old elev and azim of the plot if there. So the new plot starts with the same view
        try:
            elev = self.ax.elev
            azim = self.ax.azim
        except AttributeError:
            pass

        self.ax = self.fig.gca(projection='3d')

        rcParams['legend.fontsize'] = 11    # legend font size
        
        self.ax.set_xlabel('X axis (cm)')
        self.ax.set_xlim(-40,40)
        self.ax.set_ylabel('Y axis (cm)')
        self.ax.set_ylim(-40, 40)
        self.ax.set_zlabel('Z axis (cm)')
        self.ax.set_zlim(0, 80)

        if 'elev' in locals():
            self.ax.view_init(elev=elev, azim=azim)             # use old camera elevation and angle 

        else:
            self.ax.view_init(elev=15, azim=-100)             # set camera elevation and angle to default

        #plot the QR code at centered at 0,0
        for i,_ in enumerate(QRStart_x):
            self.ax.plot([QRStart_x[i], QREnd_x[i]], [QRStart_y[i],QREnd_y[i]])


        for i,_ in enumerate(self.camera_positions):

            #Draw the QR code 
            CamStart_x = [self.camera_positions[i][0]]
            CamEnd_x=[self.camera_pointsAt[i][0]]
            CamStart_y = [self.camera_positions[i][1]]
            CamEnd_y=[self.camera_pointsAt[i][1]]
            CamStart_z = [self.camera_positions[i][2]]
            CamEnd_z=[self.camera_pointsAt[i][2]]
            
            camera_render = self.camera_render[i]

            for j,_ in enumerate(CamStart_y):
                
                if i == self.index:
                    color = phonecolor = "#c41111"
                    alpha = 1
                else:
                    color = "#a81e1e"
                    phonecolor = "#913838"
                    alpha = 0.2

                
                #Draw the active one as red. the rest gray
                #Draw the line from the camera position to where it is pointing.
                self.ax.plot([CamStart_x[j], CamEnd_x[j]], [CamStart_y[j], CamEnd_y[j]],zs=[CamStart_z[j], CamEnd_z[j]], color = color, alpha = alpha)
                    
                #Draw the camera. A rectangle representing an iPhone6                      
                self.ax.plot([camera_render[0][0], camera_render[1][0]], [camera_render[0][1], camera_render[1][1]], zs=[camera_render[0][2], camera_render[1][2]], color = phonecolor, alpha = alpha)
                self.ax.plot([camera_render[1][0], camera_render[2][0]], [camera_render[1][1], camera_render[2][1]], zs=[camera_render[1][2], camera_render[2][2]], color = phonecolor, alpha = alpha)
                self.ax.plot([camera_render[2][0], camera_render[3][0]], [camera_render[2][1], camera_render[3][1]], zs=[camera_render[2][2], camera_render[3][2]], color = phonecolor, alpha = alpha)
                self.ax.plot([camera_render[3][0], camera_render[0][0]], [camera_render[3][1], camera_render[0][1]], zs=[camera_render[3][2], camera_render[0][2]], color = phonecolor, alpha = alpha)


def main ():

    global options, args, root

    if options.filename:
        # if individual filename provided
        assert os.path.exists(options.filename), "File " + options.filename + " could not be read"
        filenames = [options.filename]

    else:
        #else load all jpgs in imgs folder
        filenames = [filename for filename in glob.glob("imgs/*.JPG")+glob.glob("imgs/*.jpg")]

    
    cvimages, tkimages, image_sizes                  = loadImages(filenames)
    corner_points                                    = getCornersOfQRCode(cvimages)
    camera_positions, camera_pointsAt, camera_render = getCameraLocationInfo(corner_points, image_sizes)

    print("ready")

    gui = tkGUI(root,tkimages,camera_positions,camera_pointsAt,camera_render)
    Tk.mainloop() 


                
if __name__ == '__main__':
    try:
         
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=globals()['__doc__'])
        parser.add_option("-f", "--file", dest="filename", help="filename of the image to read", metavar="FILE")


        (options, args) = parser.parse_args()
        root = Tk.Tk()
        main()
        

        sys.exit(0)
    except KeyboardInterrupt as e: # Ctrl-C
        raise e
    except SystemExit as e: # sys.exit()
        raise e
    except Exception as e:
        print ('ERROR, UNEXPECTED EXCEPTION')
        print (str(e))
        traceback.print_exc()
        os._exit(1)