import os
import numpy as np
# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import torch
from ultralytics import YOLO

class HumanDetector(Node):
	# ノード名
	SELFNODE = "human_detector"
	def __init__(self):
		# ノードの初期化
		super().__init__(self.SELFNODE)
		self.get_logger().info("%s initializing..." % (self.SELFNODE))
		# モデルの読み込み
		model_name = self.param("model_name", 'yolov8x-pose.pt').string_value # yolov8n.pt,yolov8x-seg.pt
		self.model = YOLO(model_name)
		# ros2 init
		self.human_image_pub_ = self.create_publisher(Image,'human_detector/human_image', 1)
		self.mask_image_pub_ = self.create_publisher(Image,'human_detector/mask_image', 1)
		self.resaul_image_pub_ = self.create_publisher(Image,'human_detector/resaul_image', 1)
		self.image_sub_ = self.create_subscription(Image,'image_raw', self.image_callback, qos_profile=ReliabilityPolicy.RELIABLE)
		self.image_ = None
		self.bridge_ = CvBridge()

	def __del__(self):
		self.get_logger().info("%s done." % self.SELFNODE)


	def param(self, name, value):
		self.declare_parameter(name, value)
		return self.get_parameter(name).get_parameter_value()

	def image_callback(self, msg):
		# print("image_callback")
		self.image_ = msg
		# if self.image_ is None:
		# 	return
		raw_image = self.bridge_.imgmsg_to_cv2(self.image_)
		img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
		# 人物検出
		results = self.model.track(source=img, show=False, save=True)
		# get predict result
		human_image = None
		human_mask = np.zeros(img.shape, dtype = np.uint8)
		for result in results:
			boxes = result.boxes  # Boxes object for bbox outputs
			masks = result.masks  # Masks object for segmentation masks outputs
			names = result.names
			orig_img = result.orig_img
			orig_shape = result.orig_shape
			speed = result.speed
			i=0
			print(f"Number of boxes = {len(boxes)}")
			print("-----")
			for box in boxes:
				ids = box.id
				pos = box.xyxy[0]
				conf = box.conf[0].item()
				cls = int(box.cls[0].item())
				clsnm = names[cls]
				keypoint = result.keypoints.xy[i]
				i += 1
				x0 = int(pos[0].item())
				y0 = int(pos[1].item())
				x1 = int(pos[2].item())
				y1 = int(pos[3].item())
				print(f"+++++ Box-{i}")
				print(f"class = {cls}:{clsnm}"); # 検知アイテムのクラス
				print(f"conf = {conf:.5f}"); # 検知アイテムの信頼度
				print(f"id = {ids}"); # 検知アイテムのID
				print(f"position = X0:{x0} Y0:{y0} X1:{x1} Y1:{y1}"); # 検知アイテムのPosition
				print("+++++")
				cv2.rectangle(human_mask, (x0, y0), (x1, y1), (255, 255, 255), -1)
				kc = 0
			print ("masks:",masks)
			print ("speed:",speed)

			# Display the annotated frame
		if len(results[0]) > 0:
			results_frame = results[0].plot(line_width=5, font_size=1)
			human_image = cv2.bitwise_and(img, human_mask)

		img_msg = self.bridge_.cv2_to_imgmsg(results_frame, encoding='bgr8')
		img_msg.header = self.image_.header
		self.resaul_image_pub_.publish(img_msg)
		if human_image is not None:
			img_msg = self.bridge_.cv2_to_imgmsg(human_image, encoding='bgr8')
			img_msg.header = self.image_.header
			self.human_image_pub_.publish(img_msg)
			img_msg = self.bridge_.cv2_to_imgmsg(human_mask, encoding='8UC3')
			img_msg.header = self.image_.header
			self.mask_image_pub_.publish(img_msg)

def main(args=None):
	try:
		rclpy.init(args=args)
		node=HumanDetector()
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		# 終了処理
		rclpy.shutdown()


if __name__ == '__main__':
	main()