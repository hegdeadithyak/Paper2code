import cv2
import matplotlib.pyplot as plt


def conv2d(input, filter):
    input_height, input_width = len(input), len(input[0])
    filter_height, filter_width = len(filter), len(filter[0])

    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    output = [[0] * output_width for _ in range(output_height)]

    for i in range(output_height):
        for j in range(output_width):
            for k in range(filter_height):
                for l in range(filter_width):
                    output[i][j] += input[i + k][j + l] * filter[k][l]

    return output


# Test the function
image_path = "/home/adithyahegdekota/Documents/GitHub/PaperReplica/THINGS_FROM_SCRATCH/AmritaCoimbattore.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the filter
filter = [[1, 0], [0, 1]]

# Apply the convolution
output = conv2d(image, filter)

# Plot the output
plt.imshow(output, cmap="gray")
plt.show()
