// Independent experimental source. Project licensing remains undecided.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rosidl_buffer/buffer.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using Image = sensor_msgs::msg::Image;
static_assert(std::is_same_v<decltype(Image{}.data), rosidl::Buffer<uint8_t>>,
              "Require the ROSIDL buffer-backed standard Image schema");
constexpr uint32_t width = 1920, height = 1080, step = width * 3 + 16;
constexpr uint64_t frames = 30;
uint8_t pixel(uint32_t y, uint32_t x, uint32_t c, uint64_t sequence) {
  return static_cast<uint8_t>((y * 17 + x * 13 + c * 7 + sequence) % 251);
}

class Probe : public rclcpp::Node {
 public:
  explicit Probe(bool publish) : Node(publish ? "cpg_probe_publisher" : "cpg_probe_subscriber"),
      publish_(publish), started_(std::chrono::steady_clock::now()) {
    if (publish_) publisher_ = create_publisher<Image>("cpg_probe/image", rclcpp::QoS(4).reliable());
    else subscription_ = create_subscription<Image>("cpg_probe/image", rclcpp::QoS(4).reliable(),
        [this](Image::ConstSharedPtr msg) { receive(msg); });
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { tick(); });
  }
  bool passed() const { return passed_; }
 private:
  void fail(const std::string &reason) {
    std::cerr << "probe_failed: " << reason << '\n';
    rclcpp::shutdown();
  }
  void tick() {
    if (std::chrono::steady_clock::now() - started_ > std::chrono::seconds(30)) {
      fail("bounded probe deadline reached"); return;
    }
    if (!publish_) return;
    if (count_ == frames) {
      if (++drain_ticks_ == 10) {
        passed_ = true;
        std::cout << "{\"status\":\"passed\",\"role\":\"publisher\",\"frames\":" << count_ << "}\n";
        rclcpp::shutdown();
      }
      return;
    }
    if (publisher_->get_subscription_count() == 0) return;
    Image msg;
    msg.height = height; msg.width = width; msg.step = step;
    msg.encoding = "rgb8"; msg.is_bigendian = 0;
    msg.header.frame_id = "cpg_probe_" + std::to_string(count_);
    msg.data.resize(static_cast<size_t>(height) * step);
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x)
        for (uint32_t c = 0; c < 3; ++c) msg.data[y * step + x * 3 + c] = pixel(y, x, c, count_);
      for (uint32_t x = width * 3; x < step; ++x) msg.data[y * step + x] = 239;
    }
    msg.header.stamp = now();
    publisher_->publish(msg); ++count_;
  }
  void receive(const Image::ConstSharedPtr &msg) {
    if (msg->width != width || msg->height != height || msg->step != step || msg->encoding != "rgb8" ||
        msg->is_bigendian != 0 || msg->data.size() != static_cast<size_t>(height) * step ||
        msg->header.frame_id != "cpg_probe_" + std::to_string(count_)) {
      fail("metadata, sequence, or payload-size mismatch"); return;
    }
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x)
        for (uint32_t c = 0; c < 3; ++c)
          if (msg->data[y * step + x * 3 + c] != pixel(y, x, c, count_)) {
            fail("pixel mismatch"); return;
          }
      for (uint32_t x = width * 3; x < step; ++x)
        if (msg->data[y * step + x] != 239) { fail("row padding mismatch"); return; }
    }
    const double latency = (now() - rclcpp::Time(msg->header.stamp)).seconds() * 1000;
    if (latency < 0) { fail("negative clock interval"); return; }
    latency_ms_.push_back(latency);
    if (++count_ == frames) {
      passed_ = true;
      std::cout << "{\"status\":\"passed\",\"role\":\"subscriber\",\"mode\":\"cpu-image\",\"frames\":"
                << count_ << ",\"width\":" << width << ",\"height\":" << height << ",\"step\":" << step
                << ",\"drops\":0,\"payload_bytes_per_frame\":" << static_cast<size_t>(height) * step
                << ",\"latency_scope\":\"publish timestamp through CPU payload validation\",\"latency_samples_ms\":[";
      for (size_t i = 0; i < latency_ms_.size(); ++i) std::cout << (i ? "," : "") << latency_ms_[i];
      std::cout << "]}\n";
      rclcpp::shutdown();
    }
  }
  bool publish_, passed_ = false;
  uint64_t count_ = 0, drain_ticks_ = 0;
  std::chrono::steady_clock::time_point started_;
  std::vector<double> latency_ms_;
  rclcpp::Publisher<Image>::SharedPtr publisher_;
  rclcpp::Subscription<Image>::SharedPtr subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc, char **argv) {
  try {
    if (argc != 2 || (std::string(argv[1]) != "publish" && std::string(argv[1]) != "subscribe"))
      throw std::invalid_argument("Require publish or subscribe");
    rclcpp::init(argc, argv);
    auto probe = std::make_shared<Probe>(std::string(argv[1]) == "publish");
    rclcpp::spin(probe);
    return probe->passed() ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
