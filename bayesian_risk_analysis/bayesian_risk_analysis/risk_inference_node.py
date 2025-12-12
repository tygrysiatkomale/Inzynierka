import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from std_msgs.msg import Int32

import numpy as np
import pickle
import os
import math
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory


class RiskInferenceNode(Node):
    def __init__(self):
        super().__init__('risk_inference_node')

        pkg_share = get_package_share_directory('bayesian_risk_analysis')
        model_path = os.path.join(pkg_share, 'models', 'integrity_network.pkl')

        self.bn_infer = None

        try:
            with open(model_path, 'rb') as f:
                self.bn_infer = pickle.load(f)
            self.get_logger().info(f"Model loaded from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model! {e}")

        self.last_pose_pos = None
        self.last_pose_time = None

        self.current_speed_gps = 0.0
        self.current_speed_filter = 0.0
        self.current_roll = 0.0
        self.current_acc_total = 9.81

        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/gnss_pose', self.pose_callback, 10) 
        self.create_subscription(Vector3Stamped, '/filter/velocity', self.vel_callback, 10)

        self.status_pub = self.create_publisher(Int32, '/integrity/system_status', 10)

        self.create_timer(0.2, self.run_inference)

    def imu_callback(self, msg):
        try:
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            w = Rotation.from_quat(q)
            euler = Rotation.as_euler('xyz', degrees=True)
            self.current_roll = abs(euler[0])
        except:
            pass

        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        self.current_acc_total = np.sqrt(ax**2 + ay**2 + az**2)

    def vel_callback(self, msg):
        vx = msg.vector.x
        vy = msg.vector.y
        vz = msg.vector.z
        self.current_speed_filter = np.sqrt(vx**2 + vy**2 + vz**2)

    def pose_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9

        current_lat = msg.pose.position.x
        current_lon = msg.pose.position.y

        if self.last_pose_pos is not None and self.last_pose_time is not None:
            dt = current_time - self.last_pose_time

            if dt > 0.001:
                METERS_PER_DEG_LAT = 111132.0
                METERS_PER_DEG_LON = 111132.0 * np.cos(np.radians((self.last_pose_pos[0] + current_lat) / 2.0))

                d_lat_m = (current_lat - self.last_pose_pos[0]) * METERS_PER_DEG_LAT
                d_lon_m = (current_lon - self.last_pose_pos[1]) * METERS_PER_DEG_LON
                
                dist = np.sqrt(d_lat_m**2 + d_lon_m**2)
                
                self.current_speed_gps = dist / dt
        
        self.last_pose_pos = (current_lat, current_lon)
        self.last_pose_time = current_time

    def run_inference(self):
        if self.bn_infer is None: 
            self.get_logger().error(f"Model is not loaded")
            return
        
        mismatch = abs(self.current_speed_filter - self.current_speed_gps)
        if mismatch < 0.2: mismatch_state = 0
        elif mismatch < 0.5: mismatch_state = 1
        else: mismatch_state = 2

        if self.current_roll < 5: roll_state = 0
        elif self.current_roll < 15: roll_state = 1
        else: roll_state = 2

        vib_diff = abs(self.current_acc_total - 9.81)
        if vib_diff < 0.5: vib_state = 0
        elif vib_diff < 2.0: vib_state = 1
        else: vib_state = 2

        try:
            evidence = {
                'Mismatch_State': mismatch_state,
                'Roll_State': roll_state,
                'Vibration_State': vib_state
            }

            q = self.bn_infer.query(variables=['System_Status'], evidence=evidence, show_progress=False)
            probs = q.values

            most_likely_state = int(np.argmax(probs))

            msg = Int32()
            msg.data = most_likely_state
            self.status_pub.publish(msg)

            self.get_logger().info(f"Input: Mis={mismatch:.2f} Roll={self.current_roll:.1f} -> Status: {most_likely_state}")

        except Exception as e:
            self.get_logger().error(f"Reasoning error: {e}")

        

def main(args=None):
    rclpy.init(args=args)
    node = RiskInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print('Hi from bayesian_risk_analysis.')


if __name__ == '__main__':
    main()
