import dummy_dataset
import matplotlib.pyplot as plt
from matplotlib import patches

image, bboxes, labels = dummy_dataset.DummyObjectsDataset.get_image_with_labels(100, 100)

# image shape: (image_height, image_width, 3)
# bboxes shape: (n_boxes, 4) --> (x0, y0, x1, y1)
# labels shape: (n_boxes, 1) --> int

fig, ax = plt.subplots(1, 1)
ax.imshow(image)
for b, l in zip(bboxes, labels):
    x0, y0, x1, y1 = b
    rec = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=0)
    ax.add_patch(rec)
    print(f"box: {b}, label id: {l}, label str: {dummy_dataset.dataset.ShapeTypes.int_to_string(l)}")

plt.show()
