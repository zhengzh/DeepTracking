import sys
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import numpy as np
import rosbag

from tqdm import tqdm

def generate_data(bag_name, num=-1):
    
    # data is too large, only online generate data
    # speed, translation, rotation augment data
    
    angle_min, angle_max, angle_increment = None, None, None

    ranges = []

    count = 0
    bag = rosbag.Bag('./data/%s.bag' % (bag_name))

    total_msgs = bag.get_message_count(topic_filters='/scan')

    for topic, msg, t in tqdm(bag.read_messages(topics='/scan'), total=total_msgs):
        # TODO tf information
        if not angle_min:
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            angle_increment = msg.angle_increment
            print("angle_min, max, inc %f, %f, %f" % (angle_min, angle_max, angle_increment))

        if count % 2 == 0: # 20hz, sample 2, 4, 5
            ranges.append(msg.ranges)

        count += 1
        
        if num > 0 and len(ranges) >= num:
            break
            
    rg = np.array(ranges)
    rg = np.where(rg > 0.01, rg, 50.) # hokuyo give 0.001 when maximum range
    
    return rg


if __name__ == "__main__":
    bag_name = 'shitang'
    rg = generate_data(bag_name)

    print('saving...')
    np.save('./data/%s' % (bag_name), rg)


