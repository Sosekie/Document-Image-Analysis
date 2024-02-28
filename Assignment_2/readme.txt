Name: Chenrui
Surname: Fan

Link: https://github.com/Sosekie/Document-Image-Analysis/tree/main/Assignment_2

Description:

Main idea is first using PIL to load image and using numpy to convert it.
For gray-scale image, it is not just the average of RGB but with grayscale = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2].
For RGB histogram, call function one by one and plot them on the subplot.
For representing the histogram of a color image, use plt.fill_between to plot, set transparent range(alpha) to make it more visible.