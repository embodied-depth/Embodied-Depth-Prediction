import numpy as np
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from random import randint


class CollisionError(Exception):
    def __init__(self, error_info=None):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class Controller:
    def __init__(self):
        rospy.init_node('myagent')
        self.rate = rospy.Rate(20)
        
        self.ACTION_SPACE = {
                'move_forward': Twist(
                    Vector3(-0.1, 0, 0),
                    Vector3(0, 0, 0),            
                    ),
                'back': Twist(
                    Vector3(0.1, 0, 0),
                    Vector3(0, 0, 0),            
                    ),
                'turn_left': Twist(
                    Vector3(0., 0, 0),
                    Vector3(0, 0, 0.1),            
                    ),
                'turn_right': Twist(
                    Vector3(0., 0, 0),
                    Vector3(0, 0, -0.1),            
                    ),
        }

        rospy.Subscriber("/scan", LaserScan, self.lasercallback)
        self.pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.collision_flag = False
        
    def act(self ,policy:list):
        if not self.collision_flag:
            for a in policy:
                if not self.collision_flag:
                    self._act(a)
                else:
                    break
        if self.collision_flag:
            for a in ['back']*5:
                self._act(a)

    def _act(self, action:str):
        rospy.loginfo(action)
        if not self.collision_flag:
            steps = 10 if action == 'move_forward' else 15
            for i in range(steps): # enlong the move time
                self.pub.publish(self.ACTION_SPACE[action])
                self.rate.sleep()
        else: 
            self.pub.publish(self.ACTION_SPACE['back'])
    
    def lasercallback(self, msg):
        arr =  np.array(msg.ranges[100:600]) # 120 degree
        arr = arr[arr > 0.03] # filter numbers below the min range
        self.collision_flag = (arr < 0.25).any()

def test_control():
    pub = Controller()
    #while not rospy.is_shutdown():
    for _ in range(10):
        action = ['move_forward'] * 5
        pub.act(action)

def lasercallback(msg):
    arr =  np.array(msg.ranges[180:540]) # 90 degree
    arr = arr[arr > 0.03] # filter numbers below the min range
    return (arr < 0.2).any()
    
def baselidar_test():
    rospy.init_node('scan_values')
    rospy.Subscriber("/scan",LaserScan, lasercallback)
    rospy.spin()

if __name__ == '__main__':
    #baselidar_test()
    test_control()