import numpy as np
import math

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis



def make_lidar_map(pose_x, pose_y, pose_theta, curr_map, lidar_sensor_readings, display):
    
    print(pose_x, pose_y, pose_theta, lidar_sensor_readings)
    # print(lidar_sensor_readings)
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y
        
        # print(rx, ry)÷s

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            
            #try catch for bounds just in case
            try:
                grey_val = min(curr_map[359-abs(int(wx*30))][abs(int(wy*30))] + .005, 1.0)
                curr_map[359-abs(int(wx*30))][abs(int(wy*30))] = grey_val
                # You will eventually REPLACE the following lines with a more robust version of the map
                # with a grayscale drawing containing more levels than just 0 and 1.
                
                color = (grey_val*256**2+grey_val*256+grey_val)*255
                display.setColor(int(color))
                display.drawPixel(360-abs(int(wx*30)),abs(int(wy*30)))
            except:
                pass
                
            

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))