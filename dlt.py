import numpy as N 

def Normalization(nd,x):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Inputs:
     nd: number of dimensions (2 for 2D; 3 for 3D)
     x: the data to be normalized (directions at different columns and points at rows)
    Outputs:
     Tr: the transformation matrix (translation plus scaling)
     x: the transformed data
    '''

    x = N.asarray(x)
    m, s = N.mean(x,0), N.std(x)
    if nd==2:
        Tr = N.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = N.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
        
    Tr = N.linalg.inv(Tr)
    x = N.dot( Tr, N.concatenate( (x.T, N.ones((1,x.shape[0]))) ) )
    x = x[0:nd,:].T

    return Tr, x



def DLTcalib(xyz, uv, nd=3):
    '''
    Camera calibration by DLT using known object points and their image points.

    This code performs 2D or 3D DLT camera calibration with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    Inputs:
     nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
     xyz are the coordinates in the object 3D or 2D space of the calibration points.
     uv are the coordinates in the image 2D space of these calibration points.
     The coordinates (x,y,z and u,v) are given as columns and the different points as rows.
     For the 2D DLT (object planar space), only the first 2 columns (x and y) are used.
     There must be at least 6 calibration points for the 3D DLT and 4 for the 2D DLT.
    Outputs:
     L: array of the 8 or 11 parameters of the calibration matrix
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    
    #Convert all variables to numpy array:
    xyz = N.asarray(xyz)
    uv = N.asarray(uv)
    #number of points:
    np = xyz.shape[0]
    #Check the parameters:

    #Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
    #This is relevant when there is a considerable perspective distortion.
    #Normalization: mean position at origin and mean distance equals to 1 at each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []
    if nd == 2: #2D DLT
        for i in range(np):
            x,y = xyzn[i,0], xyzn[i,1]
            u,v = uvn[i,0], uvn[i,1]
            A.append( [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] )
            A.append( [0, 0, 0, x, y, 1, -v*x, -v*y, -v] )
    elif nd == 3: #3D DLT
        for i in range(np):
            x,y,z = xyzn[i,0], xyzn[i,1], xyzn[i,2]
            u,v = uvn[i,0], uvn[i,1]
            A.append( [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u] )
            A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v] )

    #convert A to array
    A = N.asarray(A) 
    #Find the 11 (or 8 for 2D DLT) parameters:
    U, S, Vh = N.linalg.svd(A)
    #The parameters are in the last line of Vh and normalize them:
    L = Vh[-1,:] / Vh[-1,-1]
    #Camera projection matrix:
    H = L.reshape(3,nd+1)
    #Denormalization:
    H = N.dot( N.dot( N.linalg.pinv(Tuv), H ), Txyz );
    H = H / H[-1,-1]
    L = H.flatten(0)
    #Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = N.dot( H, N.concatenate( (xyz.T, N.ones((1,xyz.shape[0]))) ) ) 
    uv2 = uv2/uv2[2,:] 
    #mean distance:
    err = N.sqrt( N.mean(N.sum( (uv2[0:2,:].T - uv)**2,1 )) ) 




    return L, err, H

def project(xyz, H):
    uv2 = N.dot( H, N.concatenate( (xyz.T, N.ones((1,xyz.shape[0]))) ) ) 
    uv2 = uv2/uv2[2,:] 

    return uv2[0:2,:].T