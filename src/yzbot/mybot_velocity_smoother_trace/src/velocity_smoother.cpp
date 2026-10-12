// Copyright (c) 2022 Samsung Research
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "nav2_velocity_smoother/velocity_smoother.hpp"

using namespace std::chrono_literals;
using nav2_util::declare_parameter_if_not_declared;
using std::placeholders::_1;
using rcl_interfaces::msg::ParameterType;

namespace
{
int64_t steadyNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Minimal JSON string escaper for the read-only transport path field.
std::string jsonString(const std::string & value)
{
  std::string out = "\"";
  for (const char c : value) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += '?';
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

// Read-only scope guard: covers *all* stock timer exits (including timeout
// returns), so a busy timer can be distinguished from a quick timeout return.
struct TimerDurationScope
{
  std::atomic<int64_t> & last;
  std::atomic<int64_t> & maximum;
  std::atomic<int64_t> & inflight_start;
  std::atomic<uint64_t> & completed;
  const int64_t begin = steadyNs();

  TimerDurationScope(
    std::atomic<int64_t> & last_arg, std::atomic<int64_t> & maximum_arg,
    std::atomic<int64_t> & inflight_arg, std::atomic<uint64_t> & completed_arg)
  : last(last_arg), maximum(maximum_arg), inflight_start(inflight_arg),
    completed(completed_arg)
  {
    inflight_start.store(begin, std::memory_order_relaxed);
  }

  ~TimerDurationScope()
  {
    const auto duration = steadyNs() - begin;
    auto prior = maximum.load(std::memory_order_relaxed);
    while (duration > prior && !maximum.compare_exchange_weak(
        prior, duration, std::memory_order_relaxed)) {}
    last.store(duration, std::memory_order_relaxed);
    completed.fetch_add(1, std::memory_order_relaxed);
    inflight_start.store(0, std::memory_order_relaxed);
  }
};
}  // namespace

namespace nav2_velocity_smoother
{

VelocitySmoother::VelocitySmoother(const rclcpp::NodeOptions & options)
: LifecycleNode("velocity_smoother", "", options),
  last_command_time_{0, 0, get_clock()->get_clock_type()}
{
  // An experimental binary must never run without a persisted trace. Stock
  // Nav2 uses a different package and is completely unaffected.
  const char * path = std::getenv("MYBOT_SMOOTHER_TRACE_FILE");
  if (!path || path[0] != '/') {
    throw std::runtime_error("MYBOT_SMOOTHER_TRACE_FILE must be an absolute path");
  }
  trace_file_ = path;
  // Read-only record of which process-local DDS profile this smoother got, so
  // a UDP-transport round can be told apart from a default-transport one
  // without trusting any external observer.
  if (const char * profile = std::getenv("FASTRTPS_DEFAULT_PROFILES_FILE")) {
    trace_transport_profile_ = profile;
  }
  // Opt-in P0 discriminator, disabled unless set to a positive number of
  // seconds. It only ever runs while the normal callback has already gone quiet.
  if (const char * fallback = std::getenv("MYBOT_SMOOTHER_POLL_FALLBACK_SEC")) {
    // Empty/blank/"0" all mean "not requested", so launches and shell defaults
    // can pass the variable unconditionally. This is a diagnostic probe, so an
    // unusable value must not take the whole navigation stack down with it: a
    // blanked-out or out-of-range setting simply leaves the probe off and warns.
    std::string value(fallback);
    const auto first = value.find_first_not_of(" \t");
    if (first != std::string::npos) {
      value = value.substr(first);
      const auto last = value.find_last_not_of(" \t");
      value = value.substr(0, last + 1);
      if (value != "0") {
        try {
          const double seconds = std::stod(value);
          if (seconds > 0.0 && seconds < 60.0) {
            poll_fallback_after_ = seconds;
          } else {
            RCLCPP_WARN(
              get_logger(),
              "MYBOT_SMOOTHER_POLL_FALLBACK_SEC '%s' outside (0, 60); probe disabled",
              value.c_str());
          }
        } catch (const std::exception &) {
          RCLCPP_WARN(
            get_logger(),
            "MYBOT_SMOOTHER_POLL_FALLBACK_SEC '%s' is not a number; probe disabled",
            value.c_str());
        }
      }
    }
  }
  trace_poll_armed_ = poll_fallback_after_ > 0.0 ? 1 : 0;
  std::ofstream check(trace_file_, std::ios::app);
  if (!check) {
    throw std::runtime_error("Cannot create smoother trace file: " + trace_file_);
  }
}

VelocitySmoother::~VelocitySmoother()
{
  stopTrace();
  if (timer_) {
    timer_->cancel();
    timer_.reset();
  }
}

nav2_util::CallbackReturn
VelocitySmoother::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Configuring velocity smoother");
  auto node = shared_from_this();
  std::string feedback_type;
  double velocity_timeout_dbl;

  // Smoothing metadata
  declare_parameter_if_not_declared(node, "smoothing_frequency", rclcpp::ParameterValue(20.0));
  declare_parameter_if_not_declared(
    node, "feedback", rclcpp::ParameterValue(std::string("OPEN_LOOP")));
  declare_parameter_if_not_declared(node, "scale_velocities", rclcpp::ParameterValue(false));
  node->get_parameter("smoothing_frequency", smoothing_frequency_);
  node->get_parameter("feedback", feedback_type);
  node->get_parameter("scale_velocities", scale_velocities_);

  // Kinematics
  declare_parameter_if_not_declared(
    node, "max_velocity", rclcpp::ParameterValue(std::vector<double>{0.50, 0.0, 2.5}));
  declare_parameter_if_not_declared(
    node, "min_velocity", rclcpp::ParameterValue(std::vector<double>{-0.50, 0.0, -2.5}));
  declare_parameter_if_not_declared(
    node, "max_accel", rclcpp::ParameterValue(std::vector<double>{2.5, 0.0, 3.2}));
  declare_parameter_if_not_declared(
    node, "max_decel", rclcpp::ParameterValue(std::vector<double>{-2.5, 0.0, -3.2}));
  node->get_parameter("max_velocity", max_velocities_);
  node->get_parameter("min_velocity", min_velocities_);
  node->get_parameter("max_accel", max_accels_);
  node->get_parameter("max_decel", max_decels_);

  for (unsigned int i = 0; i != 3; i++) {
    if (max_decels_[i] > 0.0) {
      throw std::runtime_error(
              "Positive values set of deceleration! These should be negative to slow down!");
    }
    if (max_accels_[i] < 0.0) {
      throw std::runtime_error(
              "Negative values set of acceleration! These should be positive to speed up!");
    }
    if (min_velocities_[i] > max_velocities_[i]) {
      throw std::runtime_error(
              "Min velocities are higher than max velocities!");
    }
  }

  // Get feature parameters
  declare_parameter_if_not_declared(node, "odom_topic", rclcpp::ParameterValue("odom"));
  declare_parameter_if_not_declared(node, "odom_duration", rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(
    node, "deadband_velocity", rclcpp::ParameterValue(std::vector<double>{0.0, 0.0, 0.0}));
  declare_parameter_if_not_declared(node, "velocity_timeout", rclcpp::ParameterValue(1.0));
  node->get_parameter("odom_topic", odom_topic_);
  node->get_parameter("odom_duration", odom_duration_);
  node->get_parameter("deadband_velocity", deadband_velocities_);
  node->get_parameter("velocity_timeout", velocity_timeout_dbl);
  velocity_timeout_ = rclcpp::Duration::from_seconds(velocity_timeout_dbl);

  if (max_velocities_.size() != 3 || min_velocities_.size() != 3 ||
    max_accels_.size() != 3 || max_decels_.size() != 3 || deadband_velocities_.size() != 3)
  {
    throw std::runtime_error(
            "Invalid setting of kinematic and/or deadband limits!"
            " All limits must be size of 3 representing (x, y, theta).");
  }

  // Get control type
  if (feedback_type == "OPEN_LOOP") {
    open_loop_ = true;
  } else if (feedback_type == "CLOSED_LOOP") {
    open_loop_ = false;
    odom_smoother_ = std::make_unique<nav2_util::OdomSmoother>(node, odom_duration_, odom_topic_);
  } else {
    throw std::runtime_error("Invalid feedback_type, options are OPEN_LOOP and CLOSED_LOOP.");
  }

  // Setup inputs / outputs
  smoothed_cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel_smoothed", 1);
  cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", rclcpp::QoS(1),
    std::bind(&VelocitySmoother::inputCommandCallback, this, std::placeholders::_1));

  trace_lifecycle_ = 1;
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
VelocitySmoother::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Activating");
  smoothed_cmd_pub_->on_activate();
  double timer_duration_ms = 1000.0 / smoothing_frequency_;
  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(timer_duration_ms)),
    std::bind(&VelocitySmoother::smootherTimer, this));

  dyn_params_handler_ = this->add_on_set_parameters_callback(
    std::bind(&VelocitySmoother::dynamicParametersCallback, this, _1));

  // create bond connection
  createBond();
  trace_lifecycle_ = 2;
  startTrace();
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
VelocitySmoother::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Deactivating");
  trace_lifecycle_ = 1;
  stopTrace();
  if (timer_) {
    timer_->cancel();
    timer_.reset();
  }
  smoothed_cmd_pub_->on_deactivate();
  dyn_params_handler_.reset();

  // destroy bond connection
  destroyBond();
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
VelocitySmoother::on_cleanup(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Cleaning up");
  trace_lifecycle_ = 0;
  smoothed_cmd_pub_.reset();
  odom_smoother_.reset();
  cmd_sub_.reset();
  return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn
VelocitySmoother::on_shutdown(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(get_logger(), "Shutting down");
  return nav2_util::CallbackReturn::SUCCESS;
}

void VelocitySmoother::inputCommandCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  trace_input_enter_.fetch_add(1, std::memory_order_relaxed);
  // If message contains NaN or Inf, ignore
  if (!nav2_util::validateTwist(*msg)) {
    trace_input_invalid_.fetch_add(1, std::memory_order_relaxed);
    RCLCPP_ERROR(get_logger(), "Velocity message contains NaNs or Infs! Ignoring as invalid!");
    return;
  }

  command_ = msg;
  last_command_time_ = now();
  trace_input_ros_ns_ = last_command_time_.nanoseconds();
  trace_input_wz_mrad_ = static_cast<int32_t>(
    std::clamp(msg->angular.z, -1e6, 1e6) * 1000.0);
  trace_input_vx_mmps_ = static_cast<int32_t>(
    std::clamp(msg->linear.x, -1e6, 1e6) * 1000.0);
  trace_input_mono_ns_ = steadyNs();
  trace_input_valid_.fetch_add(1, std::memory_order_relaxed);
}

// Opt-in, STRICTLY DIAGNOSTIC: on a stalled tick, ask the reader directly
// whether it still holds samples the executor never dispatched.
//
// It deliberately does NOT feed what it takes back into the control path. An
// earlier version called inputCommandCallback() with the recovered sample; a
// static review showed that is unsafe, because the callback re-stamps
// last_command_time_ (line ~301), which cancels the zero-Twist timeout stop and
// can restart the base on a command up to a minute old whose staleness is
// undetectable (Twist carries no header and MessageInfo is not consulted).
// Discarding the sample instead can only ever leave the smoother in the state it
// was already in - it never makes the base move. `poll_taken` therefore answers
// "was there a sample in the reader?" without touching motion semantics.
void VelocitySmoother::pollStalledInput()
{
  trace_poll_attempts_.fetch_add(1, std::memory_order_relaxed);
  geometry_msgs::msg::Twist msg;
  rclcpp::MessageInfo info;
  uint64_t taken = 0;
  try {
    // KEEP_LAST depth 1 means at most one sample per writer is ever waiting, so
    // this exits on the first failed take. The 8 bound only guarantees the timer
    // cannot spin and lose its 20 Hz cadence.
    while (taken < 8 && cmd_sub_->take(msg, info)) {
      ++taken;
      trace_poll_last_vx_mmps_ = static_cast<int32_t>(
        std::clamp(msg.linear.x, -1e6, 1e6) * 1000.0);
      trace_poll_last_wz_mrad_ = static_cast<int32_t>(
        std::clamp(msg.angular.z, -1e6, 1e6) * 1000.0);
    }
  } catch (const rclcpp::exceptions::RCLError & error) {
    // The executor wraps its own take in a try/catch; a bare take here would
    // otherwise escape smootherTimer(), skip the publish, and leave spin().
    trace_poll_errors_.fetch_add(1, std::memory_order_relaxed);
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Poll fallback take() failed: %s", error.what());
    return;
  }
  if (taken > 0) {
    trace_poll_taken_.fetch_add(taken, std::memory_order_relaxed);
    trace_poll_last_mono_ns_ = steadyNs();
  }
}

double VelocitySmoother::findEtaConstraint(
  const double v_curr, const double v_cmd, const double accel, const double decel)
{
  // Exploiting vector scaling properties
  double dv = v_cmd - v_curr;

  double v_component_max;
  double v_component_min;

  // Accelerating if magnitude of v_cmd is above magnitude of v_curr
  // and if v_cmd and v_curr have the same sign (i.e. speed is NOT passing through 0.0)
  // Decelerating otherwise
  if (abs(v_cmd) >= abs(v_curr) && v_curr * v_cmd >= 0.0) {
    v_component_max = accel / smoothing_frequency_;
    v_component_min = -accel / smoothing_frequency_;
  } else {
    v_component_max = -decel / smoothing_frequency_;
    v_component_min = decel / smoothing_frequency_;
  }

  if (dv > v_component_max) {
    return v_component_max / dv;
  }

  if (dv < v_component_min) {
    return v_component_min / dv;
  }

  return -1.0;
}

double VelocitySmoother::applyConstraints(
  const double v_curr, const double v_cmd,
  const double accel, const double decel, const double eta)
{
  double dv = v_cmd - v_curr;

  double v_component_max;
  double v_component_min;

  // Accelerating if magnitude of v_cmd is above magnitude of v_curr
  // and if v_cmd and v_curr have the same sign (i.e. speed is NOT passing through 0.0)
  // Decelerating otherwise
  if (abs(v_cmd) >= abs(v_curr) && v_curr * v_cmd >= 0.0) {
    v_component_max = accel / smoothing_frequency_;
    v_component_min = -accel / smoothing_frequency_;
  } else {
    v_component_max = -decel / smoothing_frequency_;
    v_component_min = decel / smoothing_frequency_;
  }

  return v_curr + std::clamp(eta * dv, v_component_min, v_component_max);
}

void VelocitySmoother::smootherTimer()
{
  TimerDurationScope duration_scope{
    trace_timer_duration_last_ns_, trace_timer_duration_interval_max_ns_,
    trace_timer_inflight_start_ns_, trace_timer_duration_completed_};
  trace_timer_mono_ns_ = steadyNs();
  trace_timer_ros_ns_ = now().nanoseconds();
  trace_timer_enter_.fetch_add(1, std::memory_order_relaxed);
  // Opt-in and strictly observational: it runs before the timeout branch so the
  // sample is still in the reader, but it never writes to the control path.
  if (poll_fallback_after_ > 0.0 && command_ &&
    (now() - last_command_time_).seconds() > poll_fallback_after_)
  {
    pollStalledInput();
  }
  // Wait until the first command is received
  if (!command_) {
    trace_timer_no_command_.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  auto cmd_vel = std::make_unique<geometry_msgs::msg::Twist>();

  // Check for velocity timeout. If nothing received, publish zeros to apply deceleration
  if (now() - last_command_time_ > velocity_timeout_) {
    trace_timeout_age_ros_ns_ = (now() - last_command_time_).nanoseconds();
    if (last_cmd_ == geometry_msgs::msg::Twist() || stopped_) {
      stopped_ = true;
      trace_timer_timeout_stopped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    trace_timer_timeout_decelerate_.fetch_add(1, std::memory_order_relaxed);
    *command_ = geometry_msgs::msg::Twist();
  }

  stopped_ = false;

  // Get current velocity based on feedback type
  geometry_msgs::msg::Twist current_;
  if (open_loop_) {
    current_ = last_cmd_;
  } else {
    current_ = odom_smoother_->getTwist();
  }

  // Apply absolute velocity restrictions to the command
  command_->linear.x = std::clamp(command_->linear.x, min_velocities_[0], max_velocities_[0]);
  command_->linear.y = std::clamp(command_->linear.y, min_velocities_[1], max_velocities_[1]);
  command_->angular.z = std::clamp(command_->angular.z, min_velocities_[2], max_velocities_[2]);

  // Find if any component is not within the acceleration constraints. If so, store the most
  // significant scale factor to apply to the vector <dvx, dvy, dvw>, eta, to reduce all axes
  // proportionally to follow the same direction, within change of velocity bounds.
  // In case eta reduces another axis out of its own limit, apply accel constraint to guarantee
  // output is within limits, even if it deviates from requested command slightly.
  double eta = 1.0;
  if (scale_velocities_) {
    double curr_eta = -1.0;

    curr_eta = findEtaConstraint(
      current_.linear.x, command_->linear.x, max_accels_[0], max_decels_[0]);
    if (curr_eta > 0.0 && std::fabs(1.0 - curr_eta) > std::fabs(1.0 - eta)) {
      eta = curr_eta;
    }

    curr_eta = findEtaConstraint(
      current_.linear.y, command_->linear.y, max_accels_[1], max_decels_[1]);
    if (curr_eta > 0.0 && std::fabs(1.0 - curr_eta) > std::fabs(1.0 - eta)) {
      eta = curr_eta;
    }

    curr_eta = findEtaConstraint(
      current_.angular.z, command_->angular.z, max_accels_[2], max_decels_[2]);
    if (curr_eta > 0.0 && std::fabs(1.0 - curr_eta) > std::fabs(1.0 - eta)) {
      eta = curr_eta;
    }
  }

  cmd_vel->linear.x = applyConstraints(
    current_.linear.x, command_->linear.x, max_accels_[0], max_decels_[0], eta);
  cmd_vel->linear.y = applyConstraints(
    current_.linear.y, command_->linear.y, max_accels_[1], max_decels_[1], eta);
  cmd_vel->angular.z = applyConstraints(
    current_.angular.z, command_->angular.z, max_accels_[2], max_decels_[2], eta);
  last_cmd_ = *cmd_vel;

  // Apply deadband restrictions & publish
  cmd_vel->linear.x = fabs(cmd_vel->linear.x) < deadband_velocities_[0] ? 0.0 : cmd_vel->linear.x;
  cmd_vel->linear.y = fabs(cmd_vel->linear.y) < deadband_velocities_[1] ? 0.0 : cmd_vel->linear.y;
  cmd_vel->angular.z = fabs(cmd_vel->angular.z) <
    deadband_velocities_[2] ? 0.0 : cmd_vel->angular.z;

  trace_publish_wz_mrad_ = static_cast<int32_t>(
    std::clamp(cmd_vel->angular.z, -1e6, 1e6) * 1000.0);
  trace_publish_vx_mmps_ = static_cast<int32_t>(
    std::clamp(cmd_vel->linear.x, -1e6, 1e6) * 1000.0);
  trace_publish_mono_ns_ = steadyNs();
  trace_publish_before_.fetch_add(1, std::memory_order_relaxed);
  smoothed_cmd_pub_->publish(std::move(cmd_vel));
  trace_publish_after_.fetch_add(1, std::memory_order_relaxed);
}

rcl_interfaces::msg::SetParametersResult
VelocitySmoother::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  for (auto parameter : parameters) {
    const auto & type = parameter.get_type();
    const auto & name = parameter.get_name();

    if (type == ParameterType::PARAMETER_DOUBLE) {
      if (name == "smoothing_frequency") {
        smoothing_frequency_ = parameter.as_double();
        if (timer_) {
          timer_->cancel();
          timer_.reset();
        }

        double timer_duration_ms = 1000.0 / smoothing_frequency_;
        timer_ = this->create_wall_timer(
          std::chrono::milliseconds(static_cast<int>(timer_duration_ms)),
          std::bind(&VelocitySmoother::smootherTimer, this));
      } else if (name == "velocity_timeout") {
        velocity_timeout_ = rclcpp::Duration::from_seconds(parameter.as_double());
      } else if (name == "odom_duration") {
        odom_duration_ = parameter.as_double();
        odom_smoother_ =
          std::make_unique<nav2_util::OdomSmoother>(
          shared_from_this(), odom_duration_, odom_topic_);
      }
    } else if (type == ParameterType::PARAMETER_DOUBLE_ARRAY) {
      if (parameter.as_double_array().size() != 3) {
        RCLCPP_WARN(get_logger(), "Invalid size of parameter %s. Must be size 3", name.c_str());
        result.successful = false;
        break;
      }

      if (name == "max_velocity") {
        max_velocities_ = parameter.as_double_array();
      } else if (name == "min_velocity") {
        min_velocities_ = parameter.as_double_array();
      } else if (name == "max_accel") {
        for (unsigned int i = 0; i != 3; i++) {
          if (parameter.as_double_array()[i] < 0.0) {
            RCLCPP_WARN(
              get_logger(),
              "Negative values set of acceleration! These should be positive to speed up!");
            result.successful = false;
          }
        }
        max_accels_ = parameter.as_double_array();
      } else if (name == "max_decel") {
        for (unsigned int i = 0; i != 3; i++) {
          if (parameter.as_double_array()[i] > 0.0) {
            RCLCPP_WARN(
              get_logger(),
              "Positive values set of deceleration! These should be negative to slow down!");
            result.successful = false;
          }
        }
        max_decels_ = parameter.as_double_array();
      } else if (name == "deadband_velocity") {
        deadband_velocities_ = parameter.as_double_array();
      }
    } else if (type == ParameterType::PARAMETER_STRING) {
      if (name == "feedback") {
        if (parameter.as_string() == "OPEN_LOOP") {
          open_loop_ = true;
          odom_smoother_.reset();
        } else if (parameter.as_string() == "CLOSED_LOOP") {
          open_loop_ = false;
          odom_smoother_ =
            std::make_unique<nav2_util::OdomSmoother>(
            shared_from_this(), odom_duration_, odom_topic_);
        } else {
          RCLCPP_WARN(
            get_logger(), "Invalid feedback_type, options are OPEN_LOOP and CLOSED_LOOP.");
          result.successful = false;
          break;
        }
      } else if (name == "odom_topic") {
        odom_topic_ = parameter.as_string();
        odom_smoother_ =
          std::make_unique<nav2_util::OdomSmoother>(
          shared_from_this(), odom_duration_, odom_topic_);
      }
    }
  }

  return result;
}

void VelocitySmoother::startTrace()
{
  stopTrace();
  std::ofstream check(trace_file_, std::ios::app);
  if (!check) {
    throw std::runtime_error("Cannot reopen smoother trace file: " + trace_file_);
  }
  // Pass this already-open stream to the writer. Reopening inside the thread
  // would introduce a startup TOCTOU and could silently lose the first row.
  trace_running_ = true;
  trace_thread_ = std::thread(&VelocitySmoother::traceLoop, this, std::move(check));
}

void VelocitySmoother::stopTrace()
{
  trace_running_ = false;
  if (trace_thread_.joinable()) {
    trace_thread_.join();
  }
}

void VelocitySmoother::traceLoop(std::ofstream out)
{
  while (trace_running_) {
    const int64_t mono = steadyNs();
    const int64_t wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    const int64_t timer_inflight = trace_timer_inflight_start_ns_.load();
    out << std::fixed << std::setprecision(9)
        << "{\"mono_abs\":" << static_cast<double>(mono) / 1e9
        << ",\"wall\":" << static_cast<double>(wall) / 1e9
        << ",\"pid\":" << ::getpid()
        << ",\"transport_profile\":"
        << (trace_transport_profile_.empty() ? std::string("null")
            : jsonString(trace_transport_profile_))
        << ",\"lifecycle\":" << trace_lifecycle_.load()
        << ",\"input_enter\":" << trace_input_enter_.load()
        << ",\"input_valid\":" << trace_input_valid_.load()
        << ",\"input_invalid\":" << trace_input_invalid_.load()
        << ",\"poll_attempts\":" << trace_poll_attempts_.load()
        << ",\"poll_taken\":" << trace_poll_taken_.load()
        << ",\"poll_errors\":" << trace_poll_errors_.load()
        << ",\"poll_armed\":" << trace_poll_armed_.load()
        << ",\"poll_last_mono_ns\":" << trace_poll_last_mono_ns_.load()
        << ",\"poll_last_vx_mmps\":" << trace_poll_last_vx_mmps_.load()
        << ",\"poll_last_wz_mrad\":" << trace_poll_last_wz_mrad_.load()
        << ",\"timer_enter\":" << trace_timer_enter_.load()
        << ",\"timer_duration_last_ns\":" << trace_timer_duration_last_ns_.load()
        << ",\"timer_duration_interval_max_ns\":" <<
      trace_timer_duration_interval_max_ns_.exchange(0)
        << ",\"timer_duration_completed\":" << trace_timer_duration_completed_.load()
        << ",\"timer_inflight_age_ns\":" <<
      (timer_inflight > 0 ? std::max<int64_t>(0, mono-timer_inflight) : 0)
        << ",\"timer_no_command\":" << trace_timer_no_command_.load()
        << ",\"timer_timeout_stopped\":" << trace_timer_timeout_stopped_.load()
        << ",\"timer_timeout_decelerate\":" << trace_timer_timeout_decelerate_.load()
        << ",\"publish_before\":" << trace_publish_before_.load()
        << ",\"publish_after\":" << trace_publish_after_.load()
        << ",\"input_mono_ns\":" << trace_input_mono_ns_.load()
        << ",\"timer_mono_ns\":" << trace_timer_mono_ns_.load()
        << ",\"publish_mono_ns\":" << trace_publish_mono_ns_.load()
        << ",\"input_ros_ns\":" << trace_input_ros_ns_.load()
        << ",\"timer_ros_ns\":" << trace_timer_ros_ns_.load()
        << ",\"timeout_age_ros_ns\":" << trace_timeout_age_ros_ns_.load()
        << ",\"input_wz_mrad\":" << trace_input_wz_mrad_.load()
        << ",\"publish_wz_mrad\":" << trace_publish_wz_mrad_.load()
        << ",\"input_vx_mmps\":" << trace_input_vx_mmps_.load()
        << ",\"publish_vx_mmps\":" << trace_publish_vx_mmps_.load()
        << "}\n" << std::flush;
    if (!out) {
      RCLCPP_ERROR(get_logger(), "Smoother internal trace write failed: %s", trace_file_.c_str());
      trace_running_ = false;
      break;
    }
    for (int i = 0; i < 10 && trace_running_; ++i) {
      std::this_thread::sleep_for(100ms);
    }
  }
}

}  // namespace nav2_velocity_smoother

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nav2_velocity_smoother::VelocitySmoother)
