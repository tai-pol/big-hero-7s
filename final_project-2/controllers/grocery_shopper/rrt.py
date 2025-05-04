import numpy as np
import matplotlib.pyplot as plt
import math
import random

###############################################################################
## Base Code
###############################################################################
class Node:
    """
    Node for RRT/RRT* Algorithm. This is what you'll make your graph with!
    """
    def __init__(self, pt, parent=None):
        self.point = pt # n-Dimensional point
        self.parent = parent # Parent node
        self.path_from_parent = [] # List of points along the way from the parent node (for visualization)
        self.cost = 0.0 # length of path from start node to itself

frontiers = [] # is explored and has at least one neighbor that is unexplored
unknown = [] # map value of 0
explored = [] # map value of 1
obstacles = [] # map value of 2
node_list = [] # list of nodes being navigated to in the map
current_position = ()

# for testing purposes - NOT FINISHED YET!
def visualize_2D_graph(state_bounds, obstacles, nodes, goal_point=None, filename=None):
    '''
    @param state_bounds Array of min/max for each dimension
    @param obstacles Locations and radii of spheroid obstacles
    @param nodes List of vertex locations
    @param edges List of vertex connections
    '''

    fig = plt.figure()
    plt.xlim(state_bounds[0,0], state_bounds[0,1])
    plt.ylim(state_bounds[1,0], state_bounds[1,1])

    if (len(unknown) > len(explored)):
        plt.gca().set_facecolor('grey')
        for pt in explored:
            plt.plot(pt[0], pt[1], 'o', color='white', markersize=1)
    else:
        plt.gca().set_facecolor('white')
        for pt in unknown:
            plt.plot(pt[0], pt[1], 'o', color='grey', markersize=1)

    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'ro', markersize=1)

    goal_node = None
    for node in nodes:
        if node.parent is not None:
            node_path = np.array(node.path_from_parent)
            plt.plot(node_path[:,0], node_path[:,1], '-b')
        if goal_point is not None and np.linalg.norm(node.point - np.array(goal_point)) <= 1e-5:
            goal_node = node
            plt.plot(node.point[0], node.point[1], 'k^')
        else:
            plt.plot(node.point[0], node.point[1], 'ro')

    plt.plot(nodes[0].point[0], nodes[0].point[1], 'ko')

    if goal_node is not None:
        cur_node = goal_node
        while cur_node is not None: 
            if cur_node.parent is not None:
                node_path = np.array(cur_node.path_from_parent)
                plt.plot(node_path[:,0], node_path[:,1], '--y')
                cur_node = cur_node.parent
            else:
                break

    if goal_point is not None:
        plt.plot(goal_point[0], goal_point[1], 'gx')


    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()

def map_update(map):
    global frontiers, unknown, explored, obstacles

    frontiers.clear()
    unknown.clear()
    explored.clear()
    obstacles.clear()
    rows, cols = map.shape
    for i in range(rows):
        for j in range(cols):
            if map[i, j] == 1: # if the current element is explored
                explored.append((i, j))

                # get valid neighbors
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        neighbori, neighborj = i+di, j+dj
                        if di == 0 and dj == 0:
                            continue
                        if 0 <= neighbori < rows and 0 <= neighborj < cols:
                            neighbors.append(map[neighbori, neighborj])

                if 0 in neighbors: # if a neighbor is unknown
                    frontiers.append((i, j))
            if map[i, j] == 0: # if the current element is unknown
                unknown.append((i, j))
            if map[i, j] == 2: # if the current element is an obstacle
                obstacles.append((i, j))

    return frontiers, unknown, explored, obstacles
    
def get_random_frontier_vertex():
    if len(frontiers) > 0:
        return frontiers[np.random.randint(0, len(frontiers)-2)]

def get_random_frontier_vertex_ahead(robot_map_x, robot_map_y, robot_theta):
    ang_cutoff_start = 45
    ang_cutoff_end = 135

    frontiers_ahead = []
    
    for pt in frontiers:
        angle = math.atan2(pt[1] - robot_map_y, pt[0] - robot_map_x)
        angle = (angle - robot_theta + math.pi) % (2 * math.pi) - math.pi
        
        if angle >= math.radians(ang_cutoff_start) and angle <= math.radians(ang_cutoff_end):
            frontiers_ahead.append(pt)

    if len(frontiers_ahead) == 0:
        return (0,0), False
    
    return frontiers_ahead[np.random.randint(0, len(frontiers_ahead)-2)], True

# returns the world angle theta that the vector between the two last waypoints create
def get_last_waypoint_direction(p1, p2):
    return  math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def get_random_valid_vertex():
    return explored[np.random.randint(0, len(explored)-2)]

def point_is_valid(point, map: np.array):
    '''
    @param point: n-dimensional array representing a point, should be a set of integers
    @param map: 2D array that holds the map values
    @return bool value that represents whether the point is valid (a free space) or not
    '''
    x = point[0]
    y = point[1]
    if map[x, y] == 1: # if the point is an explored, free space
        return True
    else:
        return False

# END BASE CODE
#####################################


def get_nearest_vertex(node_list, q_point):
    '''
    @param node_list: List of Node objects
    @param q_point: n-dimensional array representing a point, should be a set of integers
    @return Node in node_list with closest node.point to query q_point
    '''

    min_node = None
    min_dist = math.inf

    for node in node_list:
        distance = np.linalg.norm(np.array(node.point)-np.array(q_point))

        if distance < min_dist:
            min_dist = distance
            min_node = node

    return min_node

    raise NotImplementedError


def steer(from_point, to_point, delta_q):
    '''
    @param from_point: n-Dimensional array (point) where the path to "to_point" is originating from (e.g., [1.,2.])
    @param to_point: n-Dimensional array (point) indicating destination (e.g., [0., 0.])
    @param delta_q: Max path-length to cover, possibly resulting in changes to "to_point" (e.g., 0.2)
    @returns path: list of points leading from "from_point" to "to_point" (inclusive of endpoints)  (e.g., [ [1.,2.], [1., 1.], [0., 0.] ])
    '''

    path = []

    from_point = np.array(from_point)
    to_point = np.array(to_point)
    dist = np.linalg.norm(to_point-from_point)

    if dist > delta_q:
        to_point = from_point + ((to_point - from_point)/dist)*delta_q
    
    path = np.linspace(from_point, to_point, 10, True)

    path = [np.round(point).astype(int) for point in path]

    # for point in path:
    #     point = np.round(point, 0) # rounds the point to the nearest (x, y) indices in the map - needed for checking obstacles later

    return path

def get_nearby_nodes(node_list, q_point, radius):
    q_point = np.array(q_point)
    points = np.array([node.point for node in node_list])
    dists = np.linalg.norm(points - q_point, axis=1)
    
    nearby_indices = np.where(dists <= radius)[0]
    return [node_list[i] for i in nearby_indices]


# use this method to retrieve waypoints for the robot to navigate to a frontier
def rrt_star(map, state_bounds, obstacles, point_is_valid, starting_point, goal_point, k, delta_q):
    '''
    @param state_bounds: matrix of min/max values for each dimension (e.g., [[0,1],[0,1]] for a 2D 1m by 1m square)
    @param state_is_valid: function that maps states (N-dimensional Real vectors) to a Boolean (indicating free vs. forbidden space)
    @param k: Number of points to sample
    @param delta_q: Maximum distance allowed between vertices
    @returns List of RRT* graph nodes and the path from the starting node to the goal node
    '''

    node_list = []
    starting_node = Node(starting_point, parent=None)
    node_list.append(starting_node) # Add Node at starting point with no parent
    path_to_source = []

    for i in range(k):
        q_rand = []
        rand_num = random.random()
        if goal_point is not None and rand_num < 0.05:
            q_rand = goal_point
        else:
            q_rand = get_random_valid_vertex()
        
        # finding closest node and then calling steer function to get the location/path of new node
        q_near = get_nearest_vertex(node_list, q_rand)
        path_to_new_node = steer(q_near.point, q_rand, delta_q)
        new_node_point = path_to_new_node[len(path_to_new_node)-1]

        nearby_nodes = get_nearby_nodes(node_list, new_node_point, delta_q)

        min_cost = math.inf
        min_cost_node = None

        # finding the nearby node that has the smallest cost
        costs = np.array([node.cost for node in nearby_nodes])
        if len(costs) != 0:
            min_cost_index = np.argmin(costs)
            min_cost_node = nearby_nodes[min_cost_index]
            min_cost = min_cost_node.cost

        # if the smallest cost in nearby nodes is smaller than the original q_near cost, then make this smallest cost node the new parent of q_new
        if min_cost < q_near.cost:
            new_parent = min_cost_node
        else:
            new_parent = q_near

        # find the new path to the new parent
        path_to_new_node = steer(new_parent.point, new_node_point, delta_q)

        # checking that the node and path are in valid spaces
        if not all(point_is_valid(x, map) for x in path_to_new_node):
            continue
        
        # creating the new node and appending to node list
        q_new = Node(new_node_point, new_parent)
        q_new.path_from_parent = path_to_new_node
        q_new.cost = q_new.parent.cost + math.dist(q_new.point, q_new.parent.point)
        node_list.append(q_new)

        # rewiring the nearby neighbors with the new node
        for curr in nearby_nodes:
            # if the shortest path from start node to curr is through the new node, then rewire
            if q_new.cost + math.dist(q_new.point, curr.point) < curr.cost:
                path_to_q_new = steer(q_new.point, curr.point, delta_q)
                # check to make sure the new path is actually valid
                if all(point_is_valid(x, map) for x in path_to_q_new):
                    # if path is valid, then create the new edge from curr to new node
                    curr.parent = q_new
                    curr.path_from_parent = path_to_q_new
                    curr.cost = q_new.cost + math.dist(q_new.point, curr.point)

        # return if we're close enough to the goal point
        if goal_point is not None and math.dist(new_node_point, goal_point) <= 1e-5:
            cur_node = q_new

            # populate path_to_source with nodes from source node to q_new
            while cur_node is not starting_node: 
                path_to_source.append(cur_node.point)
                if cur_node.parent is not None:
                    cur_node = cur_node.parent
                else:
                    break

            path_to_source.reverse()
            return node_list, path_to_source

    print('DID NOT FIND PATH TO THE GOAL')
    path_to_source.reverse()
    return node_list, path_to_source

if __name__ == "__main__":
    testing_map = np.zeros(shape=[360,360])
    for i in range(90, 300):
        for j in range(90, 250):
            testing_map[i, j] = 1 #explored

    for i in range(100, 200):
        for j in range(100, 200):
            testing_map[i, j] = 2 #obstacles

    map_update(testing_map)
    starting_point = get_random_valid_vertex()
    goal_point = get_random_frontier_vertex()
    bounds = np.array([[0,360],[0,360]])
    K = 200
    nodes_rrtstar, waypoints = rrt_star(testing_map, bounds, obstacles, point_is_valid, starting_point, goal_point, K, 30)
    visualize_2D_graph(bounds, obstacles, nodes_rrtstar, goal_point, 'rrt_star_run1.png')