
# our method is smarter, safer, better, simple, efficient


import matplotlib.pyplot as plt
from builtins import round
import math
from math import pi
import numpy as np


class Params:

    def __init__(self, angle_min, angle_max, angle_increment):

        self.angle_min = angle_min
        self.angle_max = angle_max
        self.laser_step = angle_increment

        self.num_beam = round((angle_max - angle_min) /
                              angle_increment) + 1  # make it int

        self.gh = 400  # y
        self.gw = 400  # x

        self.ih = 200  # input width
        self.iw = 200  # input height

        self.g_step = 0.1
        # (x-gw/2)*g_step


params = Params(-1.500983, 1.496620, 0.004363)


# params = Params(-pi*0.7, pi*0.7, pi*0.01)

# def get_params(angle_min, angle_max, ):

#     params = object()
#     params.angle_min =

def get_lookup(params):

    gw, gh, g_step = params.gw, params.gh, params.g_step
    laser_step = params.laser_step

    dist = np.zeros((gh, gw))
    index = np.zeros((gh, gw), dtype=np.int)

    for y in range(gh):
        for x in range(gw):
            px = (x-gw/2)*g_step
            py = (y-gh/2)*g_step
            angle = math.atan2(py, px)
            # grid -> distance, beam_index
            dist[y][x] = math.sqrt(px * px + py * py)
            index[y][x] = int((pi_to_pi(angle) + pi) / laser_step)  # -pi -> 0

    return dist, index


def pi_to_pi(theta):  # note [-pi, pi) !!
    if theta >= pi:  # pi equals -pi
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta


PI_2 = 2 * math.pi
grid_dist, grid_beam_index = get_lookup(params)


def get_360_laser(laser, yaw, params):

    laser_step = params.laser_step

    n = int(PI_2 / laser_step)
    laser_360 = np.zeros(n)

    start_angle = pi_to_pi(params.angle_min + yaw) + pi
    start_index = int(start_angle / laser_step)

    index = np.arange(start_index, start_index+params.num_beam) % n

    laser_360[index] = laser

    return laser_360


def laser_to_grid(laser, gs):

    dist = laser[grid_beam_index]

    obs = np.abs(dist - grid_dist) < gs * 0.7071
    vis = dist + gs * 0.7071 > grid_dist

    return obs, vis


def get_input(obs, vis, dx, dy, ih, iw):

    gh, gw = obs.shape
    big_map = np.zeros((2, 2*gh, 2*gw))

    sh = gh//2 + dy
    sw = gw//2 + dx

    big_map[:, sh:sh+gh, sw:sw+gw] = np.stack((obs, vis))

    sih = gh-ih//2
    siw = gw-20
    # siw = gw-iw//2
    input = big_map[:, sih:sih+ih, siw:siw+iw]

    return input


def laser_to_map(laser, tx, ty, rot, params):

    g_step = params.g_step
    laser = get_360_laser(laser, rot, params)
    obs, vis = laser_to_grid(laser, g_step)
    ih, iw = params.ih, params.iw
    input = get_input(obs, vis, int(tx/g_step), int(ty/g_step), ih, iw)

    return input


def to_ogm(data):
    import rospy
    from nav_msgs.msg import OccupancyGrid
    from rospy.numpy_msg import numpy_msg

    OG = numpy_msg(OccupancyGrid)
    n, c, h, w = data.shape
    
    msgs = []
    for dat in data:
        for ch in dat:
            msg = OG()
            msg.info.width = w
            msg.info.height = h
            msg.info.resolution = 0.05
            msg.header.frame_id = 'map'
            msg.data = ch.ravel()


def show_input(input, axes):
    axes[0].imshow(input[0], 'Greys')
    axes[1].imshow(input[1], 'Greys')


def load_sensor_data(file, params):

    # [pos, laser], num, seq_len, 3|N
    data = np.load(file)
    pos, laser = data['pos'], data['laser']


def process_data(pos, laser, params):

    # pos [num, seq_len, 3]
    # laser [num, seq_len, 720]

    pos = pos - pos[:, 0].reshape((-1, 1))
    dx, dy, dyaw = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]

    n, l, _ = laser.shape

    for i in range(n):
        for j in range(l):
            laser = laser[i, j, :]
            yaw = dyaw[i, j, :]
            laser = get_360_laser(laser, yaw, params)
            obs, vis = laser_to_grid(laser, params.g_step)

    # x1-x2, y1-y2, rotate(-yaw2), yaw1-yaw2
    # reference to first laser frame
    # -> dx, dy, dyaw


def test():
    num_beam = params.num_beam
    beams = np.ones(num_beam) * 5 + 1.0
    # beams = np.random.random(num_beam) * 10 + 1.0
    angles = np.linspace(0, 2 * pi, num_beam) - pi

    laser = get_360_laser(beams, 0, params)
    obs, vis = laser_to_grid(laser, params.g_step)
    ih, iw = params.ih, params.iw
    input = get_input(obs, vis, 0, 0, ih, iw)

    return input
    plt.imshow(obs, cmap='Greys')
    plt.imshow(vis, cmap='Greys')


def main():
    pass


if __name__ == '__main__':
    test()
    # main()

    # x, y, yaw, -w/2+tep

    # laser -> grid map -> translate (visibility, obstacle) -> input
    # this map operation is very cool
    # size of map
    # center around robot?
    # transform and remove outside index, no need to rotate
    # same size
    # dt = 0.1, 100 seq
    # if we have local grid map, transform to global grid map
    # laser -> 360 laser -> grid map l translate -> input
