import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.declare_parameter('max_msgs', 20)
        self.max_msgs = self.get_parameter('max_msgs').get_parameter_value().integer_value
        self.pub = self.create_publisher(String, 'chatter', 10)
        self.count = 0
        self.timer = self.create_timer(0.5, self._tick)

    def _tick(self):
        msg = String()
        msg.data = f'Hello {self.count}'
        self.pub.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.count += 1
        if self.count >= self.max_msgs:
            self.get_logger().info('Done, shutting down.')
            self.timer.cancel()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = Talker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
