# Dummy Object Dataset

The purpose of this "tool" is that we can easily debug our Machine Learning solutions for **object detection** (in the
near future: **segmentation** and **keypoint detection**)

I always had trouble finding small datasets where my model can easily overfit, or where I can check if it works
as it is supposed to. So the best solution is: generate the simplest images in the fly.

This tool is highly influenced by: [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/shapes.py#L63)

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
import dummy_dataset as dds

data_generator = dds.DummyObjectsDataset.get_batch_data_iterator(image_height=100,
                                                                 image_width=100,
                                                                 batch_size=32)

for d in data_generator:
    batch_images, batch_bboxes, batch_labels = d
    print(batch_images.shape)
    print(batch_bboxes.shape)
    print(batch_labels.shape)

```
