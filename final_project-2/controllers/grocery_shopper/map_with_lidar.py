import numpy as np
import math

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis






def get_points_between(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """
    Returns a list of (x,y) tuples that lie on a straight line between (x0,y0) and (x1,y1)
    using Bresenham's line algorithm.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    while n > 0:
        points.append((x, y))
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
        n -= 1

    return points


def make_lidar_map(pose_x, pose_y, pose_theta, curr_map, lidar_sensor_readings, display):
    
    amount_add_per_cycle = .003
    explored_add_per_cycle = -.001
    
    # Convert robot position to display coordinates
    robot_x = (pose_x + 15) * 10
    robot_y = (pose_y + 15) * 10
    
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        wx =  math.cos(t)*rx - math.sin(t)*ry 
        wy =  math.sin(t)*rx + math.cos(t)*ry 
        
        wx *= 10
        wy *= 10
        
        wx += robot_x
        wy += robot_y
            
        try:
            grey_val = min(curr_map[int(wx)][int(wy)] + amount_add_per_cycle, 1.0)
            curr_map[int(wx)][int(wy)] = grey_val
            
            color = (grey_val*256**2+grey_val*256+grey_val)*255
            display.setColor(int(color))
            display.drawPixel(int(wx), int(wy))
        
        
            # explored_points = get_points_between(robot_x, robot_y, wx, wy)
            # for x, y in explored_points:
                
            #     grey_val = max(curr_map[x][y] + explored_add_per_cycle, -1.0)
            #     curr_map[x][y] = grey_val
                
            #     grey_val = abs(grey_val)
            #     color = (grey_val*256**2+grey_val*256+grey_val)*255
            #     display.setColor(int(color))
            #     display.drawPixel(x, y)
            
        except Exception as e:
            print(e)

                
            

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x)), abs(int(pose_y)))
    
    
    
def filter_lidar_map(lidar_map: np.ndarray):
    
    obstacle_cutoff = .8
    explored_cuttoff = -.8
    
    size_x, size_y = lidar_map.shape    
    
    new_map = np.zeros_like(lidar_map)
    
    for x in size_x:
        for y in size_y:
            val = lidar_map[x, y]
            if val > obstacle_cutoff:
                new_map[x, y] = 2
            elif val < explored_cuttoff:
                new_map[x, y] = 1
            else:
                new_map[x, y] = 0
    
    return new_map