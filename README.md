# Dummy Object Dataset

**This tool is under development and not completed - contains lot of problems**

The purpose of this "tool" is that we can easily debug our Machine Learning solutions for **object detection** (in the
near future: **segmentation** and **keypoint detection**)

I always had trouble finding small datasets where my model can easily overfit, or where I can check if it works
as it is supposed to. So the best solution is: generate the simplest images in the fly.

This tool is highly influenced by the idea in: [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/shapes.py#L63)

## Sample images

We can change the min number of shapes, max number of shapes, image size, etc...

![](/art/shapes_1.png)
![](/art/shapes_2.png)
![](/art/shapes_3.png)

## Install

- Required packages
    - OpenCV 3
    - Numpy

`python3 setup.py install` or you can do this directly from github


## Usage

```
import dummy_dataset


image, bboxes, labels = dummy_dataset.DummyObjectsDataset.get_image_with_labels(image_height, image_width)

# image shape: (image_height, image_width, 3)
# bboxes shape: (n_boxes, 4) --> (x0, y0, x1, y1)
# labels shape: (n_boxes, 1) --> int

fig, ax = plt.subplots(1, 1)
ax.imshow(image)
for b in bboxes:
    x0, y0, x1, y1 = b
    rec = patches.Rectangle((x0, y0), x1-x0, y1-y0, fill=0)
    ax.add_patch(rec)
```
