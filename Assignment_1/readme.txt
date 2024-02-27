Name: Chenrui
Surname: Fan

Link: https://github.com/Sosekie/Document-Image-Analysis/tree/main/Assignment_1

Description:

There are many ways to go about upsampling. Here I have implemented nearest neighbor interpolation as well as bilinear interpolation. Note that because of the large size of the example image, it is quite time consuming (over two hours) to assign values to the image using the Iteration method, whereas using slicing and stepping in numpy takes less than a second.

I've also considered using Transposed Convolution or Unpooling, but haven't implemented it yet.

Update:

Seems we need to do down sampling instead of up scaling, so I add this part, using numpy stride.