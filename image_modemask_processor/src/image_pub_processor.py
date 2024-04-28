#!/usr/bin/env python

import rospy
import copy
from sensor_msgs.msg import Image as RosImage, PointCloud2, CameraInfo, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import cv2
from testing_model import * 
import message_filters
import sensor_msgs.point_cloud2 as pc2
import tf
import numpy as np
from PIL import Image, ImageEnhance
from image_geometry import PinholeCameraModel
from tf import TransformListener
from geometry_msgs.msg import PointStamped
import math
import struct

# from mcap_ros1.writer import Writer
# from std_msgs.msg import Header
# from geometry_msgs.msg import Point
# from sensor_msgs.msg import PointCloud2, PointField
# from rospy import Time


class ImageProcessor:
    def __init__(self):
        rospy.init_node('point_processor_node', anonymous=True)

        # self.pnt_cld_header = Header()

        self.model = UNET(in_channels=3, out_channels=3).to(DEVICE)
        self.model_file = rospy.get_param('/image_modemask_processor/model_file', '/home/mrs/ths_workspace/src/image_modemask_processor/src/trained_model_epoch675.pth')

        self.loaded_model = load_model(self.model, self.model_file, map_location='cpu')
        self.camera_info_topic = rospy.get_param('/image_modemask_processor/camera_info_topic', '/uav11/front_rgbd/color/camera_info')
        self.image_raw_topic = rospy.get_param('/image_modemask_processor/image_raw_topic', '/uav11/front_rgbd/color/image_raw')
        self.filter_number = rospy.get_param('filter_number', '1')

        # self.point_cloud_topic = rospy.get_param('/image_modemask_processor/point_cloud_topic', '/uav1/octomap_local_vis/octomap_point_cloud_centers')
        self.uav_id = rospy.get_param('/image_modemask_processor/uav_id', 'uav11')

        self.output_topic = f'/{self.uav_id}/front_rgbd/image_classed' 
        self.filtered_output_topic = f'/{self.uav_id}/front_rgbd/image_filter_applied' 
        # self.pointcloud_output_topic = f"/{self.uav_id}/points_mapped_to_classes"


        self.loaded_model.to("cpu")  

        # self.cam_model = PinholeCameraModel()

        # self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
        #   PointField('y', 4, PointField.FLOAT32, 1),
        #   PointField('z', 8, PointField.FLOAT32, 1),
        #   PointField('rgba', 12, PointField.UINT32, 1),
        #   ]
        
        # self.listener = TransformListener()
        self.bridge = CvBridge()

        self.points_to_publish = {}

        # Subscribers
        # self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        # self.image_sub = message_filters.Subscriber(self.image_raw_topic, RosImage)
        # self.pointcloud_sub = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)
        self.image_sub = rospy.Subscriber(self.image_raw_topic, RosImage, self.image_callback)

        # Publishers
        self.image_pub = rospy.Publisher(self.output_topic, RosImage, queue_size=10)
        self.filtered_image_pub = rospy.Publisher(self.filtered_output_topic, RosImage, queue_size=10)
        # self.pntcld2_pub = rospy.Publisher(self.pointcloud_output_topic, PointCloud2, queue_size=2)

    # def camera_info_callback(self, camera_info):
    #     self.cam_model.fromCameraInfo(camera_info)

    # def image_map_shared_callback(self, data_image, pointcloud_data):

    #     # For data image -> predict classes
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data_image, "bgr8")
    #     except CvBridgeError as e:
    #         rospy.logerr(e)
    #         return

    #     cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #     pil_image = PILImage.fromarray(cv_image_rgb)

    #     classed_image_pil = predict_and_save_image(self.loaded_model, pil_image)
    #     classed_image_np = np.array(classed_image_pil)

    #     classed_image_cv = cv2.cvtColor(classed_image_np, cv2.COLOR_RGB2BGR)

    #     # Publish image with classes
        
        
    #     try:
    #         classed_image_msg = self.bridge.cv2_to_imgmsg(classed_image_cv, "bgr8")
    #         self.image_pub.publish(classed_image_msg)
    #     except CvBridgeError as e:
    #         rospy.logerr(e)

    #     # For pointcloud data -> colour the points using the octomap pointcloud  
    #     try:
    #         # points = []
    #         self.listener.waitForTransform(point_stamped.header.frame_id, self.cam_model.tf_frame, point_stamped.header.stamp, rospy.Duration(1.0))
            
    #         for data in pc2.read_points(pointcloud_data, field_names=("x", "y", "z"), skip_nans=True):
    #             point_cv = (data[0], data[1], data[2])
    #             x,y,z = point_cv[0], point_cv[1], point_cv[2]

    #             point_stamped = PointStamped()
    #             point_stamped.header.frame_id = pointcloud_data.header.frame_id
    #             point_stamped.header.stamp = pointcloud_data.header.stamp
    #             point_stamped.point.x = x
    #             point_stamped.point.y = y
    #             point_stamped.point.z = z

    #             # transform the whole pointcloud at once instead of point-by-point
    #             transformed_point = self.listener.transformPoint(self.cam_model.tf_frame, point_stamped)

    #             point_image = self.cam_model.project3dToPixel((transformed_point.point.x, transformed_point.point.y, transformed_point.point.z))
    #             # unrectify the point_image to compensate camera distortion
                
    #             if 0 <= point_image[0] < classed_image_pil.width and 0 <= point_image[1] < classed_image_pil.height and transformed_point.point.z > 0.0:
    #                 color = classed_image_pil.getpixel((int(point_image[0]), int(point_image[1])))
    #                 r = color[0]
    #                 g = color[1]
    #                 b = color[2]
    #             else:
    #                 r = 0
    #                 g = 0
    #                 b = 0

    #             a = 255 #as in alpha 

    #             rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
    #             pt = [x, y, z, rgb]
    #             key_str_point = f"{x}, {y}, {z}"
    #             self.points_to_publish[key_str_point] = pt
    #             # points.append(pt)

                
    #         points = list(self.points_to_publish.values())
    #         pcointcloud2 = pc2.create_cloud(self.pnt_cld_header, self.fields, points)
    #         pcointcloud2.header.stamp = pointcloud_data.header.stamp
    #         pcointcloud2.header.frame_id = pointcloud_data.header.frame_id
    #         self.pntcld2_pub.publish(pcointcloud2)
    #         print("Projected Points, published map, published image")
            
    #     except tf.Exception as e:
    #         rospy.logwarn("TF Exception: {}".format(str(e)))
        
    def apply_high_contrast(self, pil_image, factor):

        enhancer = ImageEnhance.Contrast(pil_image)
        return enhancer.enhance(factor)

    def apply_blur(self, pil_image, kernel_size):
        return cv2.GaussianBlur(pil_image, (kernel_size, kernel_size), 1)

    
    def change_brightness(self, pil_image, factor):

        enhancer = ImageEnhance.Brightness(pil_image)
        return enhancer.enhance(factor)
    
    

    

    def image_callback(self, data_image):
        # print("IMAGE CALLBACKk")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image_rgb)

        filtered_image_cv_set = False
        filtered_image_cv = None
        pil_image_P = None
        if self.filter_number=="1":
            # 1 High contrast pic
            ####
            contrast_factor = 2.0
            pil_image_contrast = self.apply_high_contrast(pil_image, contrast_factor)
            pil_image_P = copy.deepcopy(pil_image_contrast)
            filtered_image_contrast_np = np.array(pil_image_contrast)
            filtered_image_cv = cv2.cvtColor(filtered_image_contrast_np, cv2.COLOR_RGB2BGR)
            filtered_image_cv_set= True
            ####
        elif self.filter_number=="2":
            # 2 Darker pic
            ####
            darkness_factor = 0.5
            pil_darker_image = self.change_brightness(pil_image, darkness_factor)
            pil_image_P = copy.deepcopy(pil_darker_image)

            filtered_darker_image_np = np.array(pil_darker_image)
            filtered_image_cv = cv2.cvtColor(filtered_darker_image_np, cv2.COLOR_RGB2BGR)
            filtered_image_cv_set= True

            ####

        elif self.filter_number=="3":

            # 3 blurred pic
            ####
            kernel_size = 5
            pil_blimage_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            pil_image_blurred = self.apply_blur(pil_blimage_cv, kernel_size)

            pil_image_P = Image.fromarray(cv2.cvtColor(pil_image_blurred, cv2.COLOR_BGR2RGB))

            # filtered_image_blurred_np = np.array(pil_image_blurred)
            filtered_image_cv = copy.deepcopy(pil_image_blurred)
            filtered_image_cv_set= True

            ####

        elif self.filter_number=="4":

            # 4 Brighter pic
            ####
            brightness_factor = 1.5
            pil_brighter_image = self.change_brightness(pil_image, brightness_factor)
            pil_image_P = copy.deepcopy(pil_brighter_image)

            filtered_brighter_image_np = np.array(pil_brighter_image)
            filtered_image_cv = cv2.cvtColor(filtered_brighter_image_np, cv2.COLOR_RGB2BGR)
            filtered_image_cv_set= True

            ####
        classed_image_pil = None
        if filtered_image_cv_set:
            classed_image_pil = predict_and_save_image(self.loaded_model, pil_image_P)
        else:
            classed_image_pil = predict_and_save_image(self.loaded_model, pil_image)


        classed_image_np = np.array(classed_image_pil)

        classed_image_cv = cv2.cvtColor(classed_image_np, cv2.COLOR_RGB2BGR)

        # Publish image with classes
        try:

            classed_image_msg = self.bridge.cv2_to_imgmsg(classed_image_cv, "bgr8")
            classed_image_msg.header.stamp = rospy.Time.now()

            classed_image_msg.header.frame_id = data_image.header.frame_id
            
            self.image_pub.publish(classed_image_msg)
            # print("Published classed image")

        except CvBridgeError as e:
            rospy.logerr(e)

        # Publish filtered image
        if self.filter_number != "none" and filtered_image_cv_set:
            try:
                filtered_image_msg = self.bridge.cv2_to_imgmsg(filtered_image_cv, "bgr8")
                filtered_image_msg.header.stamp = rospy.Time.now()
                filtered_image_msg.header.frame_id = data_image.header.frame_id
                
                self.filtered_image_pub.publish(filtered_image_msg)
                # print("Published filtered image")



            except CvBridgeError as e:
                rospy.logerr(e)

        
                

if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()
        # ts = message_filters.ApproximateTimeSynchronizer([image_processor.image_sub, image_processor.pointcloud_sub], 1,0.1)
        # ts.registerCallback(image_processor.image_map_shared_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

