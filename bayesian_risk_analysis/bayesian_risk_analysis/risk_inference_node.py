import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from std_msgs.msg import Int32

import numpy as np
import pickle
import os
import math
from scipy.spatial.transform import Rotation
from ament_index_python.packages import get_package_share_directory
from collections import deque


class RiskInferenceNode(Node):
    def __init__(self):
        super().__init__('risk_inference_node')

        self.declare_parameter("timer_period_s", 0.2)
        self.declare_parameter("vibration_window", 25)
        self.declare_parameter("meters_per_deg_lat", 111132.0)
        self.declare_parameter("wind_state", 0)
        self.declare_parameter("depth_state", 0)

        timer_period = float(self.get_parameter("timer_period_s").value)
        self.vibration_window = int(self.get_parameter("vibration_window").value)
        self.meters_per_deg_lat = float(self.get_parameter("meters_per_deg_lat").value)

        pkg_share = get_package_share_directory('bayesian_risk_analysis')
        model_path = os.path.join(pkg_share, 'models', 'integrity_network.pkl')

        self.bn_infer = None

        try:
            with open(model_path, 'rb') as f:
                self.bn_infer = pickle.load(f)
            self.get_logger().info(f"Model loaded from: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model from {model_path}: {e}")

        self.last_pose_latlon = None
        self.last_pose_t = None

        self.current_speed_gps = 0.0
        self.current_speed_filter = 0.0

        self.current_roll_deg = 0.0
        self.current_turn_rate = 0.0
        self.current_acc_total = 9.81
        self.current_vibration_energy = 0.0

        self.acc_total_buffer = deque(maxlen=max(1, self.vibration_window))

        self.create_subscription(Imu, "/imu/data", self.imu_callback, 10)
        self.create_subscription(PoseStamped, "/gnss_pose", self.pose_callback, 10)
        self.create_subscription(Vector3Stamped, "/filter/velocity", self.vel_callback, 10)

        self.status_pub = self.create_publisher(Int32, "/integrity/system_status", 10)

        self.create_timer(timer_period, self.run_inference)

    def imu_callback(self, msg):
        try:
            q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            rot = Rotation.from_quat(q)
            euler = rot.as_euler('xyz', degrees=True)
            self.current_roll_deg = abs(float(euler[0]))
        except Exception:
            pass

        try:
            self.current_turn_rate = abs(float(msg.angular_velocity.z))
        except Exception:
            pass

        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        self.current_acc_total = float(np.sqrt(ax**2 + ay**2 + az**2))

        self.acc_total_buffer.append(self.current_acc_total)

        if len(self.acc_total_buffer) < 2:
            self.current_vibration_energy = 0.0
        else:
            arr = np.array(self.acc_total_buffer, dtype=np.float64)
            self.current_vibration_energy = float(np.std(arr, ddof=1))

    def vel_callback(self, msg):
        vx = msg.vector.x
        vy = msg.vector.y
        vz = msg.vector.z
        self.current_speed_filter = float(np.sqrt(vx**2 + vy**2 + vz**2))

    def pose_callback(self, msg):
        try:
            t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        except Exception:
            t = self.get_clock().now().nanoseconds / 1e9

        lat = float(msg.pose.position.x)
        lon = float(msg.pose.position.y)

        if self.last_pose_latlon is not None and self.last_pose_t is not None:
            dt = t - self.last_pose_t

            if dt > 0.001:
                last_lat, last_lon = self.last_pose_latlon
                
                lat_avg = (last_lat + lat) / 2.0
                meters_per_deg_lon = self.meters_per_deg_lat * math.cos(math.radians(lat_avg))

                d_lat_m = (lat - last_lat) * self.meters_per_deg_lat
                d_lon_m = (lon - last_lon) * meters_per_deg_lon
                
                dist = math.sqrt(d_lat_m**2 + d_lon_m**2)
                
                self.current_speed_gps = dist / dt
        
        self.last_pose_latlon = (lat, lon)
        self.last_pose_t = t

    @staticmethod
    def _clamp_int(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(x)))

    def run_inference(self):
        if self.bn_infer is None: 
            self.get_logger().error(f"Model is not loaded")
            return
        
        mismatch = abs(self.current_speed_filter - self.current_speed_gps)
        if mismatch < 0.2: mismatch_state = 0
        elif mismatch < 0.5: mismatch_state = 1
        else: mismatch_state = 2

        if self.current_roll_deg < 2.0: roll_state = 0
        elif self.current_roll_deg < 10.0: roll_state = 1
        else: roll_state = 2

        vib = self.current_vibration_energy
        if vib < 0.15: vib_state = 0
        elif vib < 0.5: vib_state = 1
        else: vib_state = 2

        turn = self.current_turn_rate
        if turn < 0.1: turn_state = 0
        elif turn < 0.3: turn_state = 1
        else: turn_state = 2

        if self.current_speed_filter < 0.2: engine_status = 0
        else: engine_status = 1

        wind_state = self._clamp_int(self.get_parameter("wind_state").value, 0, 2)
        depth_state = self._clamp_int(self.get_parameter("depth_state").value, 0, 2)

        evidence = {
            "Mismatch_State": mismatch_state,
            "Roll_State": roll_state,
            "Vibration_State": vib_state,
            "Turn_State": turn_state,
            "Engine_Status": engine_status,
            "Wind_State": wind_state,
            "Depth_State": depth_state,
            }

        try:
            q = self.bn_infer.query(variables=['System_Status'], evidence=evidence, show_progress=False)
            probs = np.array(q.values, dtype=np.float64)

            most_likely_state = int(np.argmax(probs))

            msg = Int32()
            msg.data = most_likely_state
            self.status_pub.publish(msg)

            self.get_logger().info(
                "evidence: Mis=%d Roll=%d Vib=%d Turn=%d Eng=%d Wind=%d Depth=%d | "
                "raw: mismatch=%.3f roll=%.2f vib_std=%.3f turn=%.3f v_f=%.3f v_gps=%.3f | "
                "P=[%.3f %.3f %.3f] -> state=%d"
                % (
                    mismatch_state, roll_state, vib_state, turn_state, engine_status, wind_state, depth_state,
                    mismatch, self.current_roll_deg, vib, turn, self.current_speed_filter, self.current_speed_gps,
                    probs[0], probs[1], probs[2], most_likely_state
                )
            )

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

if __name__ == '__main__':
    main()
    