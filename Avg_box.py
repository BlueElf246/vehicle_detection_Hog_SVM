# The class receives the paramenets of bounding box detection and averages box detection.
class BoundingBoxes:
	def __init__(self, nf=5):
		# defines the length of queue used to buffer data from 'nf' frames
		self.nf = nf
		# hot windows of the last n frames
		self.recent_boxes = deque([], maxlen=nf)
		# hot windows of current frame
		self.currect_boxes = None
		# all hot windows for last n frames
		self.all_boxes = []

	def update_all_boxes_(self):
		all_boxes = []
		for boxes in self.recent_boxes:
			all_boxes += boxes
		if len(all_boxes) == 0:
			self.all_boxes = []
		else:
			self.all_boxes = all_boxes

	def add(self, boxes):
		self.currect_boxes = boxes
		self.recent_boxes.appendleft(boxes)
		self.update_all_boxes_()