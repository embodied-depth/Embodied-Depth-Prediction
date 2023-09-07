import os
import glob
import rospy
import numpy as np
import cv2
import sys
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import orbslam2
import time
from collections import deque
from scipy.spatial.transform import Rotation as R

cvb=CvBridge()


input_length = 0
def slaminput(system):
    global input_length
    #if len(system.depth_buffer) > input_length and len(system.depth_buffer) == len(system.rgb_buffer):
    if len(system.depth_buffer) > input_length and len(system.rgb_buffer) > input_length:
        system.slam.process_image_rgbd(system.rgb_buffer[input_length],
                system.depth_buffer[input_length] / 1000.,
                len(system.rgb_buffer))

        input_length += 1

        if (str(system.slam.get_tracking_state()) == 'LOST'):
            system.slam_lost_count += 1
            if system.slam_lost_count > 60:
                #system.print_traj()
                print("*" * 20)
                print('RESET cuz lost too long')
                print("*" * 20)
                system.write_to_disk()
                system.reset()
                input_length = 0
                time.sleep(1)
    #if (str(system.slam.get_tracking_state()) == 'LOST'):
    #    system.slam_lost_count += 1
    #    if system.slam_lost_count > 60:
    #        #system.print_traj()
    #        print("*" * 20)
    #        print('RESET cuz lost too long')
    #        print("*" * 20)
    #        system.write_to_disk()
    #        system.reset()
    #        input_length = 0
    #        time.sleep(5)



def imnormalize(image):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : image:image data.
    : return: Numpy array of normalize data
    """
    xmax = np.max(image)
    xmin = 0
    a = 0
    b = 255

    return ((np.array(image,dtype=np.float32) - xmin) * (b - a)) / (xmax - xmin)


class ImageListener:
    def __init__(self, vocab_path, settings_path, disk_path, send_data_flag=True):
        self.disk_path = disk_path
        if not os.path.exists(self.disk_path):
            os.makedirs(self.disk_path)
        self.rgb_buffer = []
        self.depth_buffer = []
        self.odom = []
        self.i_rgb=0
        self.i_depth=0
        self.i_odom=0
        self.rgb_mem=None
        self.depth_mem=None
        self.first_flag_depth=True
        self.first_flag_rgb=True

        self.last_trans_index = 0
        self.pause_receive_flag = False
        self.static_flag = False
        self.last_static_flag = False
        self.flag_smoother = deque(maxlen=80)



        self.traj_num_idx = 0
        self.endflag = False

        self.send_data_flag = send_data_flag

        ## Receiver from realseanse
        rospy.init_node("grabrgb")
        rospy.loginfo("Running RGB Grabber")
        rospy.Subscriber("/camera/color/image_raw",Image,self.grabrgb)
        #rospy.Subscriber("/camera/depth/image_rect_raw",Image,self.checkdepth)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,self.checkdepth)
        #rospy.spin()


        ## Open or close the RGBD save by the odometry
        rospy.Subscriber('/odometry/filtered', Odometry, self.listen_odom)

        ## Slam setting
        self.slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.RGBD)
        self.slam.set_use_viewer(True)
        self.slam.initialize()
        self.slam_lost_count = 0

        ## Send to pytorch module
        self.pubimg = rospy.Publisher('data/rgb', Image,queue_size=10)
        self.pubdepth = rospy.Publisher('data/depth', Image,queue_size=10)
        self.pubtraj = rospy.Publisher('data/traj', Image,queue_size=10)
        #rospy.init_node('talker', anonymous=True)
        self.rate = rospy.Rate(20) # 10hz

        ## 

    def reset(self):
        self.traj_num_idx += 1

        self.rgb_buffer = []
        self.depth_buffer = []
        self.odom = []
        self.i_rgb=0
        self.i_depth=0
        self.i_odom=0
        self.rgb_mem=None
        self.depth_mem=None
        self.first_flag_depth=True
        self.first_flag_rgb=True

        self.static_flag = False
        self.last_static_flag = False
        self.slam.reset()
        self.slam_lost_count = 0

    def listen_odom(self, msg):
        self.last_static_flag = self.static_flag
        self.static_flag = (abs(msg.twist.twist.linear.x) < 8e-3)  and \
                (abs(msg.twist.twist.angular.z) < 8e-3 )
        self.flag_smoother.append(self.static_flag)

        if (self.i_odom <= self.i_depth) and (not self.static_flag) and (not self.pause_receive_flag):
            rmatrix = R.from_quat([msg.pose.pose.orientation.x,
                                    msg.pose.pose.orientation.y,
                                    msg.pose.pose.orientation.z,
                                    msg.pose.pose.orientation.w]).as_matrix()
            t_vec = np.array([[-msg.pose.pose.position.x],
                              [-msg.pose.pose.position.y],
                              [ msg.pose.pose.position.z]])
            zero = np.array([[0, 0, 0, 1]])
            matrix = np.concatenate([rmatrix, t_vec], axis=1)
            matrix = np.concatenate([matrix, zero], axis=0)

            self.i_odom += 1
            self.odom.append(matrix)


        if (self.last_static_flag != self.static_flag):
            print('Static flag changes from {} to {} '.format(self.last_static_flag, self.static_flag))

        if self.send_data_flag and np.array(self.flag_smoother).all() and (len(self.rgb_buffer) - self.last_trans_index) > 10:
            # If remain still for a little while, start to transmit the data
            self.send_data()


    def grabrgb(self, msg):
        #print('flag rgb :', self.pause_receive_flag)
        if (self.i_rgb <= self.i_depth) and (not self.static_flag) and (not self.pause_receive_flag):
            try:
                cv_image = cvb.imgmsg_to_cv2(msg,"bgr8")
            except CvBridgeError as e:
                print(e)

            #print(msg.step)
            image_normal= np.array(cv_image)
            if self.first_flag_rgb == True :
                self.rgb_mem = np.copy(image_normal)
                self.first_flag_rgb=False
            elif np.array_equal(self.rgb_mem,image_normal) :
                return None
            else :
                self.rgb_mem = np.copy(image_normal)

            self.i_rgb+=1
            self.rgb_buffer.append(image_normal)


            #if len(self.depth_buffer) > 0:
            #    self.slam.process_image_rgbd(self.rgb_buffer[-1],
            #            self.depth_buffer[-1] / 1000.,
            #            len(self.rgb_buffer))

            #    if (str(self.slam.get_tracking_state()) == 'LOST'):
            #        self.slam_lost_count += 1
            #        if self.slam_lost_count > 60:
            #            self.print_traj()
            #            print('RESET cuz lost too long')
            #            self.slam.reset()
            #            self.slam_lost_count = 0


    def checkdepth(self, msg):
        if (not self.static_flag) and (not self.pause_receive_flag):
            try:
                cv_image = cvb.imgmsg_to_cv2(msg,msg.encoding)
            except CvBridgeError as e:
                print(e)
            
            image_normal= np.array(imnormalize(cv_image),dtype=np.uint8)
            numpy_image= np.array(cv_image)
            if self.first_flag_depth == True:
                self.depth_mem = np.copy(numpy_image)
                self.first_flag_depth=False
            if (self.depth_mem==numpy_image).all() :
                return
            else:
                self.depth_mem = np.copy(numpy_image)
                
            self.i_depth+=1
            self.depth_buffer.append(numpy_image)

    def get_recent_traj(self):
        out = []
        trajs = self.slam.get_trajectory_points()

        print('Len of trajs: ', len(trajs))
        print('Len of rgb: ', len(self.rgb_buffer))

        for i in range(self.last_trans_index, len(trajs)):
            traj = trajs[i]
            stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
            ext_matrix = np.array([[r00, r01, r02, t0],
                                   [r10, r11, r12, t1],
                                   [r20, r21, r22, t2],
                                   [0.,  0.,  0.,  1]])
            out.append(ext_matrix)

        out = np.stack(out)
        return out

    def get_all_traj(self):
        out = []
        trajs = self.slam.get_trajectory_points()

        for i in range(0, len(trajs)):
            traj = trajs[i]
            stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
            ext_matrix = np.array([[r00, r01, r02, t0],
                                   [r10, r11, r12, t1],
                                   [r20, r21, r22, t2],
                                   [0.,  0.,  0.,  1]])
            out.append(ext_matrix)

        out = np.stack(out)
        return out

    def end(self):
        self.slam.shutdown()

        out_file_name = os.path.join( self.disk_path, 'train1_files.txt')
        shootings = glob.glob( os.path.join(self.disk_path, 'traj*'), recursive=False)

        for shoot in shootings:
            trajs = np.load(os.path.join(shoot, 'poses.npy'))
            num = trajs.shape[0]


            with open(out_file_name, 'a') as f:
                for i in range(20, num-20):
                    f.write(shoot + ' ' + str(i) + ' l\n')

        self.endflag = True

    def send_data(self):
        self.pause_receive_flag = True
        self.input_rest_rgbd2slam()
        self.align_rgb_d_len()


        if len(self.rgb_buffer) - self.last_trans_index > 0:
            traj = self.get_recent_traj()

            send_len = min(traj.shape[0] + self.last_trans_index, len(self.rgb_buffer))
            traj = traj[: send_len - self.last_trans_index]

            #for i in range(self.last_trans_index, len(self.rgb_buffer)):
            for i in range(self.last_trans_index, send_len):
                img = self.rgb_buffer[i]
                dep = self.depth_buffer[i]

                self.pubimg.publish(cvb.cv2_to_imgmsg(img))
                self.pubdepth.publish(cvb.cv2_to_imgmsg(dep))
                self.rate.sleep()


            self.last_trans_index = len(self.rgb_buffer)
            self.pubtraj.publish(cvb.cv2_to_imgmsg(traj))
            self.rate.sleep()

            print('Send Over')

        self.pause_receive_flag = False

    def write_to_disk(self):
        self.input_rest_rgbd2slam()
        self.align_rgb_d_len()
        path = os.path.join(self.disk_path, 'traj{}'.format(self.traj_num_idx))
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + '/color'):
            os.makedirs(path+ '/color')
            os.makedirs(path+ '/depth')
            os.makedirs(path+ '/depthmap')

        print('*'*20)
        print('Save into traj{}'.format(self.traj_num_idx) )
        print('Image Length: ', len(self.rgb_buffer))
        print('Depth Length: ', len(self.depth_buffer))
        traj = self.get_all_traj()
        print('Traj Length: ', traj.shape[0])
        np.save(path + '/poses.npy', traj)

        odoms = np.stack(self.odom)
        np.save(path + '/odom.npy', odoms)

        for i, (im, d) in enumerate(zip(self.rgb_buffer, self.depth_buffer)):
            #print('RGB  shape,', im.shape)
            #print('DE, shape, ', d.shape)
            cv2.imwrite(path + '/color/frame{}.jpg'.format(i), im)
            np.save(path + '/depth/dframe{}.npy'.format(i), d)
            d_ = imnormalize(d)
            cv2.imwrite(path + '/depthmap/dframe{}.jpg'.format(i), d_)

        self.pause_receive_flag = False

    # Only used for online test
    def print_traj(self):
        out = []
        trajs = self.slam.get_trajectory_points()
        i = 0
        for traj in trajs:
            stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 = traj
            ext_matrix = np.array([[r00, r01, r02, t0],
                                   [r10, r11, r12, t1],
                                   [r20, r21, r22, t2],
                                   [0.,  0.,  0.,  1]])
            i += 1
            if i % 20 == 0:
                print('Time: {}, matrix: {}'.format(stamp, ext_matrix))

    def input_rest_rgbd2slam(self):
        self.pause_receive_flag = True
        self.align_rgb_d_len()
        global input_length
        while input_length  < len(self.rgb_buffer):
            slaminput(self)

        time.sleep(2)

    def align_rgb_d_len(self):
        while len(self.rgb_buffer) > len(self.depth_buffer):
            self.rgb_buffer.pop()

        while len(self.rgb_buffer) < len(self.depth_buffer):
            self.depth_buffer.pop()

        while len(self.odom) > len(self.rgb_buffer):
            self.odom.pop()
        

def write(img, depth, path):
    print('Image Length: ', len(img))
    print('Depth Length: ', len(depth))
    for i, (im, d) in enumerate(zip(img, depth)):
        #print('RGB  shape,', im.shape)
        #print('DE, shape, ', d.shape)
        cv2.imwrite(path + '/frame{}.jpg'.format(i), im)
        #np.save(path + '/depth/dframe{}.npy'.format(i), d)
        d_ = imnormalize(d)
        cv2.imwrite(path + '/dframe{}.jpg'.format(i), d_)


def test():
    if len(sys.argv) != 4:
        print('Usage: python ./roscam.py path_to_vocabulary path_to_settings  save_rgbd_dirname' )
    lis = ImageListener(
                        vocab_path=sys.argv[1],
                        settings_path=sys.argv[2],
                        disk_path='/home/yilundu/qingquantmp/tmp/' + str(sys.argv[3])
                        )


    start_time = time.time()
    while time.time() - start_time < 40:
        slaminput(lis)

    #lis.print_traj()
    lis.write_to_disk()
    #write(lis.rgb_buffer, lis.depth_buffer, '/home/yilundu/qingquantmp/tmp/testfig')
    print('end')
    lis.end()

def main():
    if len(sys.argv) < 4:
        print('Usage: python ./roscam.py path_to_vocabulary path_to_settings  save_rgbd_dirname send_data_flag' )

    if len(sys.argv) >4:
        send_data_flag = bool(sys.argv[4])
    else:
        send_data_flag = False

    lis = ImageListener(
                        vocab_path=sys.argv[1],
                        settings_path=sys.argv[2],
                        disk_path='/home/yilundu/qingquantmp/tmp/' + str(sys.argv[3]),
                        send_data_flag=send_data_flag
                        )


    start_time = time.time()
    while time.time() - start_time < 120:
        slaminput(lis)

    lis.write_to_disk()
    lis.end()


if __name__ == '__main__':
    main()
