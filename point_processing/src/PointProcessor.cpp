
/* each ROS nodelet must have these */
#include <ros/ros.h>
#include <ros/package.h>
#include <nodelet/nodelet.h>

/* TF2 related ROS includes */
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/TransformStamped.h>

/* camera image messages */
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include "sensor_msgs/PointCloud2.h"


/* long unsigned integer message */
#include <std_msgs/UInt64.h>

/* some STL includes */
#include <stdlib.h>
#include <stdio.h>
#include <mutex>
#include <nav_msgs/Odometry.h>

/* some OpenCV includes */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tf2/LinearMath/Quaternion.h>

/* ROS includes for working with OpenCV and images */
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

/* custom helper functions from our library */
#include <mrs_lib/param_loader.h>
#include <mrs_lib/transformer.h>
#include <mrs_lib/subscribe_handler.h>
#include <mrs_lib/lkf.h>
#include <mrs_lib/geometry/misc.h>


/* for calling simple ros services */
#include <std_srvs/Trigger.h>

/* for math calculations */
#include <cmath>

/*synchronization*/
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <string>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <math.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>



#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeNode.h>
#include <octomap/OccupancyOcTreeBase.h>

#include <octomap/ColorOcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>



namespace point_processing {


class PointProcessor : public nodelet::Nodelet {

public: 
  void onInit() override;

private:

    std::atomic<bool> is_initialized_ = false; // flag*
    bool _gui_ = true; 
    std::string _uav_name_; 
    std::string raw_image_topic;
    std::string pcld_topic;
    std::string classed_image_topic;
    std::string camera_info_in;
    std::string world_origin;
    double threshold = 1e-6;
    std::string m_name_eagle_odom_msg;

    // std::unique_ptr<mrs_lib::Transformer> transformer_;
    mrs_lib::Transformer transformer_;


    tf::TransformListener tf_listener;

    // | ---------------------- msg callbacks --------------------- |

  void callbackImage(const sensor_msgs::ImageConstPtr& msg);
  
  image_transport::Subscriber sub_image_;
  
  void                               callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);
  ros::Subscriber                    sub_camera_info_;
  ros::Subscriber sub_point_cloud;


  image_geometry::PinholeCameraModel camera_model_;

  // void sharedcallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg);
  void sharedcallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg);
  // sensor_msgs::CameraInfoConstPtr

  void callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg);

  std::mutex mutex_counters_;           // to prevent data races when accessing the following variables from multiple threads
  uint64_t   image_counter_   = 0;      // counts the number of images received
  bool       got_image_       = false;  // indicates whether at least one image message was received
  bool       got_camera_info_ = false;  // indicates whether at least one camera info message was received

  // | ----------------------- publishers ----------------------- |

  ros::Publisher             pub_points_;
  ros::Publisher             colored_pointcloud_publisher;
  // ros::Publisher             marker_pub;
  ros::Publisher             octomapPub;
  std::string pub_pnt_cld_topic;
  std::map<std::tuple<int, int, int>, std::tuple<int, int, int>> colorMap;

  std::map<std::pair<int, int>, std::tuple<double, double, int, int, int>> coordinates_to_color_map;


  int                        _rate_timer_publish_;

  message_filters::Subscriber<sensor_msgs::Image> sub_shared_image_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_shared_pntcld_;
  mrs_lib::SubscribeHandler<nav_msgs::Odometry> m_subh_eagle_odom;

  // message_filters::Subscriber<sensor_msgs::CameraInfo> sub_shared_camera_info_;


  using MySyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2>;
  boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;
};


void PointProcessor::onInit() {

  got_image_       = false;
  got_camera_info_ = false;

  /* obtain node handle */
  ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

  /* waits for the ROS to publish clock */
  ros::Time::waitForValid();

  // | ------------------- load ros parameters ------------------ |
  /* (mrs_lib implementation checks whether the parameter was loaded or not) */

  mrs_lib::ParamLoader param_loader(nh, "PointProcessor");

  param_loader.loadParam("UAV_NAME", _uav_name_);
  param_loader.loadParam("gui", _gui_);
  param_loader.loadParam("camera_info_in", camera_info_in);
  param_loader.loadParam("raw_image", raw_image_topic);
  param_loader.loadParam("pcld_topic", pcld_topic);
  param_loader.loadParam("classed_image", classed_image_topic);
  param_loader.loadParam("pub_pnt_cld_topic", pub_pnt_cld_topic);
  param_loader.loadParam("world_origin", world_origin);

  ROS_INFO("My uav name: %s", _uav_name_.c_str());
  if (!param_loader.loadedSuccessfully()) {
    ROS_ERROR("[PointProcessor]: failed to load non-optional parameters!");
    ros::shutdown();
  }
  // | --------------------- tf transformer --------------------- |

  // transformer_ = std::make_unique<mrs_lib::Transformer>("PointProcessor");
  // transformer_->setDefaultPrefix(_uav_name_);
  // transformer_->retryLookupNewest(true);
  transformer_ = mrs_lib::Transformer(nh, "PointProcessor", ros::Duration(1));
  transformer_.setLookupTimeout(ros::Duration(0.1));


  /* initialize the image transport, needs node handle */
  image_transport::ImageTransport it(nh);

  // | ----------------- initialize subscribers ----------------- |
  // sub_image_       = it.subscribe(raw_image_topic, 1, &PointProcessor::callbackImage, this);

  sub_camera_info_ = nh.subscribe(camera_info_in, 1, &PointProcessor::callbackCameraInfo, this, ros::TransportHints().tcpNoDelay());

  // sub_point_cloud = nh.subscribe(pub_pnt_cld_topic, 1, &PointProcessor::callbackPointCloud, this);

  colored_pointcloud_publisher = nh.advertise<sensor_msgs::PointCloud2>(pub_pnt_cld_topic, 1);
  // marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/uav1/point_cloud_markers", 10);

  octomapPub = nh.advertise<octomap_msgs::Octomap>("/colored_octomap", 1);
  mrs_lib::SubscribeHandlerOptions shopt{nh};
  mrs_lib::construct_object(m_subh_eagle_odom,
                                  shopt,
                                  m_name_eagle_odom_msg);





  sub_shared_image_.subscribe(nh, classed_image_topic, 1);
  sub_shared_pntcld_.subscribe(nh, pcld_topic, 20);  
  // sub_shared_camera_info_.subscribe(nh, camera_info_in, 1);  



  using tt = message_filters::Synchronizer<MySyncPolicy>;
  sync_ = boost::make_shared<tt>(MySyncPolicy(4),
                                      sub_shared_image_,
                                      sub_shared_pntcld_);
  sync_->registerCallback(&PointProcessor::sharedcallback, this);


  
  ROS_INFO_ONCE("initialized==true");

  is_initialized_ = true;



}


void PointProcessor::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg) {
  if (!is_initialized_) {
    return;
  }

  got_camera_info_       = true;

  camera_model_.fromCameraInfo(*msg);
  
}

void PointProcessor::callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg){

  if (!is_initialized_) {
    return;
  }


  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::fromROSMsg(*pointcloud_msg, *cloud);

  // // Create an Octomap
  // octomap::ColorOcTree octree(0.1);  // Set the resolution of the octomap

  // // Populate Octomap with RGB values from the point cloud
  // for (const auto& point : cloud->points) {
  //     octomap::point3d endpoint(point.x, point.y, point.z);
  //     octomap::ColorOcTreeNode* node = octree.updateNode(endpoint, true);  // Create the node if it doesn't exist

  //     // Set RGB values to the Octomap node
  //     node->setColor(point.r, point.g, point.b);
  // }

  // // Convert Octomap to Octomap message
  // octomap_msgs::Octomap octomap_msg;
  // octomap_msgs::fullMapToMsg(octree, octomap_msg);

  // // Publish the Octomap message
  // octomapPub_.publish(octomap_msg);
  ROS_INFO("[PointProcessor]: OCTOMAP PUBLISHED CALLBACK");




}

void PointProcessor::callbackImage(const sensor_msgs::ImageConstPtr& msg) {

  if (!is_initialized_) {
    return;
  }

  const std::string color_encoding     = "bgr8";
  const std::string grayscale_encoding = "mono8";

  {
    std::scoped_lock lock(mutex_counters_);
    got_image_ = true;
    image_counter_++;
  }

  const cv_bridge::CvImageConstPtr bridge_image_ptr = cv_bridge::toCvShare(msg, color_encoding);
  const std_msgs::Header           msg_header       = msg->header;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, color_encoding);
  }

  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  
  cv::Mat dImg = cv_ptr->image;
  cv::Mat image = dImg;
  std::mutex m1;


  ROS_INFO_THROTTLE(1, "[PointProcessor]: Total of %u images received so far", (unsigned int)image_counter_);

}

void PointProcessor::sharedcallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg)
{
  if (!is_initialized_) {
    return;
  }

  // IMAGE PART

  const std::string color_encoding     = "bgr8";

  {
    std::scoped_lock lock(mutex_counters_);
    got_image_ = true;
    image_counter_++;
  }

  const cv_bridge::CvImageConstPtr bridge_image_ptr = cv_bridge::toCvShare(image_msg, color_encoding);
  const std_msgs::Header           msg_header       = image_msg->header;
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(image_msg, color_encoding);
  }

  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  
  cv::Mat dImg = cv_ptr->image;
  cv::Mat image = dImg;


  // POINT CLOUD PART
  ROS_INFO("[PointProcessor]: Synchronization successful");
  using PointT =pcl::PointXYZ;
  using PointRGB = pcl::PointXYZRGB;

  pcl::PointCloud<PointT>::Ptr cloud = boost::make_shared<pcl::PointCloud<PointT>>();

  //cloud = (pcl::PointCloud<PointT>::Ptr) (new pcl::PointCloud<PointT>);
  pcl::fromROSMsg (*pointcloud_msg, *cloud);
  ros::Time t = pointcloud_msg->header.stamp;
  tf::StampedTransform trans;

  try
    {
        tf_listener.waitForTransform(camera_model_.tfFrame(), pointcloud_msg->header.frame_id, t, ros::Duration(3.0));
        tf_listener.lookupTransform(camera_model_.tfFrame(), pointcloud_msg->header.frame_id, t, trans);
    }
    catch (tf::TransformException& ex){
        ROS_ERROR("%s",ex.what());
        ROS_WARN("Cannot accumulate");
        return;
  }
  ROS_INFO("[PointProcessor]: SUCCESFULL TRANSFORM IN SHARED CALLBACK");


  double fx = camera_model_.fx();  // Focal length in x direction
  double fy = camera_model_.fy();  // Focal length in y direction
  double cx = camera_model_.cx();  // Principal point in x direction
  double cy = camera_model_.cy();  // Principal point in y direction
  int x_cameras;
  int y_cameras;
  int z_cameras;

  if (m_subh_eagle_odom.hasMsg()){
    const auto msg_eagle_odom = m_subh_eagle_odom.getMsg();
    const auto new_twist_st_opt = transformer_.transformAsVector(msg_eagle_odom->child_frame_id,
                                                                  {msg_eagle_odom->twist.twist.linear.x,
                                                                    msg_eagle_odom->twist.twist.linear.y,
                                                                    msg_eagle_odom->twist.twist.linear.z},
                                                                  world_origin,
                                                                  msg_eagle_odom->header.stamp);

    if (new_twist_st_opt.has_value()) {
        x_cameras = (*new_twist_st_opt)(0, 0);
        y_cameras = (*new_twist_st_opt)(0, 1);
        z_cameras = (*new_twist_st_opt)(0, 2);
    } else {
        ROS_ERROR("[PointProcessor]]: pose or twist transformation has no value");
        return;
    }



  }
  ROS_INFO("Drone's position in world frame: x: %d, y: %d, z: %d", x_cameras, y_cameras, z_cameras);


pcl::PointCloud<PointRGB>::Ptr colored_cloud(new pcl::PointCloud<PointRGB>);


colored_cloud->width = cloud->width;
colored_cloud->height = cloud->height;
colored_cloud->is_dense = cloud->is_dense;
colored_cloud->points.resize(colored_cloud->width * colored_cloud->height);

for (size_t i = 0; i < cloud->points.size(); ++i) {



  pcl::PointXYZRGB point;
  point.x = cloud->points[i].x;
  point.y = cloud->points[i].y;
  point.z = cloud->points[i].z;
  // Set the RGB values as needed. Here's an example:
  point.r = 255; // Example: setting red color
  point.g = 255;
  point.b = 255;

  tf::Vector3 transformed_point = trans * tf::Vector3(point.x, point.y, point.z);

  if (transformed_point.z() > 0){
      //camera_model_.projectPointTo3D()
      const double pixel_x = fx * transformed_point.x() / transformed_point.z() + cx;
      const double pixel_y = fy * transformed_point.y() / transformed_point.z() + cy;
      const int pixel_x_int = static_cast<int>(std::round(pixel_x));
      const int pixel_y_int = static_cast<int>(std::round(pixel_y));

      if (pixel_x_int >= 0 && pixel_x_int < image.cols && pixel_y_int >= 0 && pixel_y_int < image.rows) {
        cv::Vec3b color = image.at<cv::Vec3b>(pixel_y_int, pixel_x_int);
        // V1
        // Print the color information V1
        // if (color[2]!=0 || color[1]!=0 || color[0]!=0 ){

          point.r = color[2];  // Red channel
          point.g = color[1];   // Green channel
          point.b = color[0]; 

        // }

      } 
      // else {

      //   auto it_cm = colorMap.find(std::make_tuple(point.x, point.y, point.z));
      //   if (it_cm != colorMap.end()) {
      //     auto& [r_retrieved, g_retrived, b_retrieved] = it_cm->second; 
      //     point.r = r_retrieved;
      //     point.g = g_retrived;
      //     point.b = b_retrieved;

      //   }
      // } 
    } 
    // else {

    //   auto it_cm = colorMap.find(std::make_tuple(point.x, point.y, point.z));
    //   if (it_cm != colorMap.end()) {
    //     auto& [r_retrieved, g_retrived, b_retrieved] = it_cm->second; 
        
    //     point.r = r_retrieved;
    //     point.g = g_retrived;
    //     point.b = b_retrieved;

    //   }  
      
    // }

  // if (point.x==x_cameras && point.y==y_cameras && point.z==z_cameras){
  //   point.r = 255;
  //   point.g = 255;
  //   point.b = 255;
  // }

  // colorMap[std::make_tuple(point.x, point.y, point.z)] = std::make_tuple(point.r, point.g, point.b);

  colored_cloud->points[i] = point;



  }



  // V2 with filter
  // for (const auto& entry : coordinates_to_color_map){

  //   ColorPointT colored_point;

  //   colored_point.x = entry.first.first;
  //   colored_point.y = entry.first.second;
  //   colored_point.z = std::get<1>(entry.second);
  //   colored_point.r = std::get<2>(entry.second);
  //   colored_point.g = std::get<3>(entry.second);
  //   colored_point.b = std::get<4>(entry.second);

  //   colored_cloud->push_back(colored_point);

  // }

  


  ROS_INFO("[PointProcessor]:  < Populated point cloud >");
  sensor_msgs::PointCloud2 colored_cloud_msg;
 
  colored_cloud_msg.fields = pointcloud_msg->fields;

  pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
  colored_cloud_msg.header = pointcloud_msg->header;
  colored_cloud_msg.header.frame_id = pointcloud_msg->header.frame_id;
  colored_cloud_msg.header.stamp = pointcloud_msg->header.stamp;


  colored_pointcloud_publisher.publish(colored_cloud_msg);
  ROS_INFO("[PointProcessor]: PUBLISHED point cloud XOXO");



}
}



/* every nodelet must include macros which export the class as a nodelet plugin */
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(point_processing::PointProcessor, nodelet::Nodelet);


