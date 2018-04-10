from keras_rcnn.datasets.malaria import load_data
from keras_rcnn.preprocessing import ObjectDetectionGenerator
from keras_rcnn.models import RPN
import numpy
import matplotlib
import keras
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#    Load data and create object detection generators:
training, validation,test = load_data()
classes = {"rbc": 1, "not":2}
generator = ObjectDetectionGenerator()
generator = generator.flow(training, classes, (224, 224), 1.0)
validation_data = ObjectDetectionGenerator()
validation_data = validation_data.flow(validation, classes, (224, 224), 1.0)
#    Create an instance of the RPN model:
image = keras.layers.Input((224, 224, 3))
model = RPN(image, classes=len(classes) + 1)
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer)
#    Train the model:
model.fit_generator(
    epochs=10,
    generator=generator,
    steps_per_epoch=1000
)
#    Predict and visualize your anchors or proposals:
example, _ = generator.next()
target_bounding_boxes, target_image, target_labels, _ = example
target_bounding_boxes = numpy.squeeze(target_bounding_boxes)
target_image = numpy.squeeze(target_image)
target_labels = numpy.argmax(target_labels, -1)
target_labels = numpy.squeeze(target_labels)
output_anchors, output_proposals, output_deltas, output_scores = model.predict(example)
output_anchors = numpy.squeeze(output_anchors)
output_proposals = numpy.squeeze(output_proposals)
output_deltas = numpy.squeeze(output_deltas)
output_scores = numpy.squeeze(output_scores)
_, axis = matplotlib.pyplot.subplots(1)
axis.imshow(target_image)
for index, label in enumerate(target_labels):
    if label == 1:
        xy = [
            target_bounding_boxes[index][0],
            target_bounding_boxes[index][1]
        ]
        w = target_bounding_boxes[index][2] - target_bounding_boxes[index][0]
        h = target_bounding_boxes[index][3] - target_bounding_boxes[index][1]
        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="g", facecolor="none")
        axis.add_patch(rectangle)
for index, score in enumerate(output_scores):
    if score > 0.95:
        xy = [
            output_anchors[index][0],
            output_anchors[index][1]
        ]
        w = output_anchors[index][2] - output_anchors[index][0]
        h = output_anchors[index][3] - output_anchors[index][1]
        rectangle = matplotlib.patches.Rectangle(xy, w, h, edgecolor="r", facecolor="none")
        axis.add_patch(rectangle)
matplotlib.pyplot.show()