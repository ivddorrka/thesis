#include <mrs_octomap_server/conversions.h>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace octomap
{

/**
 * @brief Conversion from octomap::point3d_list
 (e.g. all occupied nodes from getOccupied()) to
 * sensor_msgs::PointCloud2
 *
 * @param points
 * @param cloud
 */
void pointsOctomapToPointCloud2(const point3d_list& points, sensor_msgs::PointCloud2& cloud) {

  // make sure the channel is valid
  std::vector<sensor_msgs::PointField>::const_iterator field_iter = cloud.fields.begin(), field_end = cloud.fields.end();

  bool has_x, has_y, has_z;
  has_x = has_y = has_z = false;
  while (field_iter != field_end) {
    if ((field_iter->name == "x") || (field_iter->name == "X"))
      has_x = true;
    if ((field_iter->name == "y") || (field_iter->name == "Y"))
      has_y = true;
    if ((field_iter->name == "z") || (field_iter->name == "Z"))
      has_z = true;
    ++field_iter;
  }

  if ((!has_x) || (!has_y) || (!has_z))
    throw std::runtime_error("One of the fields xyz does not exist");

  sensor_msgs::PointCloud2Modifier pcd_modifier(cloud);
  pcd_modifier.resize(points.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

  for (point3d_list::const_iterator it = points.begin(); it != points.end(); ++it, ++iter_x, ++iter_y, ++iter_z) {
    *iter_x = it->x();
    *iter_y = it->y();
    *iter_z = it->z();
  }
}

/**
 * @brief Conversion from a sensor_msgs::PointCLoud2 to
 * octomap::Pointcloud,
 used internally in OctoMap
 *
 * @param cloud
 * @param octomapCloud
 */
void pointCloud2ToOctomap(const sensor_msgs::PointCloud2& cloud, Pointcloud& octomapCloud) {
  octomapCloud.reserve(cloud.data.size() / cloud.point_step);

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    // Check if the point is invalid
    if (!std::isnan(*iter_x) && !std::isnan(*iter_y) && !std::isnan(*iter_z))
      octomapCloud.push_back(*iter_x, *iter_y, *iter_z);
  }
}

}  // namespace octomap
