

def cv_to_gripper(x, y, z):
    #robot -y, z, -x
    shift = [0, 0, 0]
    
    
    
    x += shift[0]
    y += shift[1]
    z += shift[2]
    
    return y, z, -x