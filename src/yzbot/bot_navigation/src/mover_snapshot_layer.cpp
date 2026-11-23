// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Vector3.h"

namespace bot_navigation
{
// Simulation only: each message is the complete authoritative set of mover
// surfaces, not a general-purpose sensor cloud or evidence of free space.
class MoverSnapshotLayer : public nav2_costmap_2d::Layer
{
  using Cloud = sensor_msgs::msg::PointCloud2;
  using Points = std::vector<std::pair<double, double>>;
  using Steady = std::chrono::steady_clock;

  void onInitialize() override
  {
    auto node = node_.lock();
    if (!node) {throw std::runtime_error("snapshot costmap node expired");}
    frame_ = layered_costmap_->getGlobalFrameID();
    if (frame_ != "odom" || !layered_costmap_->isRolling()) {
      throw std::runtime_error("mover snapshot layer requires a rolling odom costmap");
    }
    declareParameter("topic", rclcpp::ParameterValue(std::string("/movers/obstacles")));
    const auto topic = node->get_parameter(getFullName("topic")).as_string();
    // This layer always provides a coherent map, including a blocked map
    // without fresh input. Humble can otherwise wait indefinitely for a
    // non-current costmap inside FollowPath instead of rejecting the motion.
    current_ = true;
    enabled_ = true;
    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group_;
    subscription_ = node->create_subscription<Cloud>(topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&MoverSnapshotLayer::receive, this, std::placeholders::_1), options);
  }

  void receive(const Cloud::ConstSharedPtr & msg)
  {
    // Bound the decoding work and reject empty/partial or unexpected layouts.
    // The simulator emits packed little-endian XYZ float32 in base_link.
    using Field = sensor_msgs::msg::PointField;
    if (msg->header.frame_id != "base_link" || msg->is_bigendian || msg->height != 1 ||
      msg->width == 0 || msg->width > 4096 || msg->point_step != 12 ||
      msg->row_step != msg->width * 12 || msg->data.size() != msg->row_step ||
      msg->fields.size() != 3)
    {return;}
    for (unsigned i = 0; i < 3; ++i) {
      const auto & f = msg->fields[i];
      if (f.name != std::string(1, "xyz"[i]) || f.offset != i * 4 ||
        f.datatype != Field::FLOAT32 || f.count != 1)
      {return;}
    }
    const rclcpp::Time stamp(msg->header.stamp, clock_->get_clock_type());
    if (!validAge(stamp)) {return;}
    Points points;
    try {
      const auto t = tf_->lookupTransform(frame_, msg->header.frame_id, stamp);
      const auto & q = t.transform.rotation;
      const auto & d = t.transform.translation;
      const tf2::Quaternion rotation(q.x, q.y, q.z, q.w);
      points.reserve(msg->width);
      for (unsigned i = 0; i < msg->width; ++i) {
        float xyz[3];
        std::memcpy(xyz, msg->data.data() + i * 12, sizeof(xyz));
        if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) {return;}
        const auto p = tf2::quatRotate(rotation, tf2::Vector3(xyz[0], xyz[1], xyz[2]));
        if (!std::isfinite(p.x()+d.x) || !std::isfinite(p.y()+d.y)) {return;}
        points.emplace_back(p.x()+d.x, p.y()+d.y);
      }
    } catch (const tf2::TransformException &) {return;}
    std::lock_guard<std::mutex> lock(mutex_);
    // On a backwards clock reset, require a newly initialized layer instead of
    // accepting a different epoch as a valid replacement of the old snapshot.
    if (stamp.nanoseconds() <= last_stamp_.nanoseconds()) {return;}
    last_stamp_ = stamp;
    last_receive_ = Steady::now();
    points_ = std::move(points);
  }

  bool validAge(const rclcpp::Time & stamp) const
  {
    if (stamp.nanoseconds() <= 0) {return false;}
    const auto age = (clock_->now() - stamp).seconds();
    return age >= -0.10 && age <= 0.50;
  }

public:
  bool isClearable() override {return false;}
  void reset() override {}  // A navigation clear may not erase a live mover.

  void updateBounds(double, double, double, double * x0, double * y0,
    double * x1, double * y1) override
  {
    // Recompose the whole small rolling window for this prototype so both the
    // old points and their old inflation are replaced after every re-center.
    auto * grid = layered_costmap_->getCostmap();
    const double x = grid->getOriginX(), y = grid->getOriginY();
    *x0 = std::min(*x0, x);
    *y0 = std::min(*y0, y);
    *x1 = std::max(*x1, x + grid->getSizeInCellsX() * grid->getResolution());
    *y1 = std::max(*y1, y + grid->getSizeInCellsY() * grid->getResolution());
    std::lock_guard<std::mutex> lock(mutex_);
    const bool fresh = validAge(last_stamp_) &&
      (Steady::now()-last_receive_) <= std::chrono::milliseconds(750);
    if (fresh != fresh_ || !reported_) {
      RCLCPP_INFO(logger_, "Mover snapshot overlay: %s",
        fresh ? "fresh snapshot" : "no fresh snapshot; blocking the local window");
      reported_ = true;
    }
    fresh_ = fresh;
    cycle_points_ = points_;
  }

  void updateCosts(nav2_costmap_2d::Costmap2D & grid, int x0, int y0, int x1, int y1) override
  {
    if (!fresh_) {
      // Stale positions are only a last observation, not a certificate of a
      // safe route. Deliberately make this simulation overlay non-traversable
      // until a fresh snapshot arrives; do not rely on current_ alone.
      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {grid.setCost(x, y, nav2_costmap_2d::LETHAL_OBSTACLE);}
      }
      return;
    }
    for (const auto & p : cycle_points_) {
      unsigned x, y;
      if (grid.worldToMap(p.first, p.second, x, y) && static_cast<int>(x) >= x0 &&
        static_cast<int>(x) < x1 && static_cast<int>(y) >= y0 && static_cast<int>(y) < y1)
      {grid.setCost(x, y, nav2_costmap_2d::LETHAL_OBSTACLE);}
    }
  }

private:
  std::string frame_;
  std::mutex mutex_;
  rclcpp::Time last_stamp_{0, 0, RCL_ROS_TIME};
  Steady::time_point last_receive_{};
  bool fresh_{false}, reported_{false};
  Points points_, cycle_points_;
  rclcpp::Subscription<Cloud>::SharedPtr subscription_;
};
}  // namespace bot_navigation

PLUGINLIB_EXPORT_CLASS(bot_navigation::MoverSnapshotLayer, nav2_costmap_2d::Layer)
