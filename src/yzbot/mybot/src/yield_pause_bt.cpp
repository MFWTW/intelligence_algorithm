// Keep the *top-level* Nav2 goal active during a mover hold while halting its
// planner/controller/recovery subtree. The final command gate independently
// blocks BOTH velocity axes: this decorator alone is not a safety stop.
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include "behaviortree_cpp_v3/bt_factory.h"
#include "behaviortree_cpp_v3/decorator_node.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

namespace mybot
{
class YieldPause : public BT::DecoratorNode
{
public:
  YieldPause(const std::string & name, const BT::NodeConfiguration & config)
  : BT::DecoratorNode(name, config)
  {
    node_ = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
    group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
    executor_.add_callback_group(group_, node_->get_node_base_interface());
    rclcpp::SubscriptionOptions options;
    options.callback_group = group_;
    sub_ = node_->create_subscription<std_msgs::msg::Bool>(
      "/competition/yield_hold", rclcpp::QoS(10).reliable(),
      [this](std_msgs::msg::Bool::ConstSharedPtr msg) { held_ = msg->data; }, options);
  }

  static BT::PortsList providedPorts() {return {};}

  BT::NodeStatus tick() override
  {
    executor_.spin_some();
    if (held_) {
      if (child_node_->status() == BT::NodeStatus::RUNNING) {
        // Nav2's action BT nodes synchronously wait for controller cancellation
        // on halt. The velocity gate continues to suppress both axes throughout.
        child_node_->halt();
      }
      return BT::NodeStatus::RUNNING;
    }
    const auto status = child_node_->executeTick();
    if (status != BT::NodeStatus::RUNNING) {
      resetChild();
    }
    return status;
  }

  void halt() override
  {
    if (child_node_->status() == BT::NodeStatus::RUNNING) {
      child_node_->halt();
    }
    setStatus(BT::NodeStatus::IDLE);
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_;
  bool held_{false};
};
}  // namespace mybot

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<mybot::YieldPause>("YieldPause");
}
