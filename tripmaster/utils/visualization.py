import io


def dot2image(dot_string, file_name=None, program="dot", format=None, return_img=False):
    """

    @param g:
    @param file_name:
    @return:
    """

    from PIL import Image

    import tempfile
    dot_file = tempfile.NamedTemporaryFile(mode='w', suffix=".dot", delete=False)
    dot_file.write(dot_string)
    dot_file.close()

    if not format:
        format = "svg"

    if not file_name and return_img:
        import tempfile
        fout = tempfile.NamedTemporaryFile(suffix="." + format)
        file_name = fout.name

    return_val = os.system(f"{program} -T {format} {dot_file.name} -o {file_name}")
    assert return_val == 0

    if return_img:
        return Image.open(file_name)


def make_square(im, size=1024, fill_color=(255, 255, 255, 255)):
    """

    Args:
        im:
        size:
        fill_color:

    Returns:

    """
    from PIL import Image
    x, y = im.size
    temp_size = max(size, x, y)
    new_im = Image.new('RGBA', (temp_size, temp_size), fill_color)
    new_im.paste(im, (int((temp_size - x) / 2), int((temp_size - y) / 2)))
    if temp_size != size:
        new_im = new_im.resize((size, size))
    return new_im


def add_title(image, title, height):
    """

    @param image:
    @type image:
    @return:
    @rtype:
    """

    # import required classes

    from PIL import Image, ImageDraw, ImageFont

    # create Image object with the input image

    image = Image.open('background.png')

    # initialise the drawing context with
    # the image object as background

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('Roboto-Bold.ttf', size=45)

    # starting position of the message

    (x, y) = (50, 50)
    message = "Happy Birthday!"
    color = 'rgb(0, 0, 0)'  # black color

    # draw the message on the background

    draw.text((x, y), message, fill=color, font=font)
    (x, y) = (150, 150)
    name = 'Vinay'
    color = 'rgb(255, 255, 255)'  # white color
    draw.text((x, y), name, fill=color, font=font)

    # save the edited image

    image.save('greeting_card.png')


def concat_images(images):
    """

    @param images:
    @type images:
    @return:
    @rtype:
    """

    from PIL import Image

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def make_pair_image(image1, image2):
    """

    @param ctb_image:
    @param oia_image:
    @return:
    """

    image1 = make_square(image1, size=2048)
    image2 = make_square(image2, size=2048)

    new_im = concat_images([image1, image2])

    return new_im


import os


def make_video(images, output_video_path, size, fps=5, is_color=True, format="XVID"):
    """

    Args:
        images:
        output_video_path:
        size:
        fps:
        is_color:
        format:

    Returns:

    """
    import cv2
    import numpy

    each_image_duration = fps  # in secs
    fourcc = cv2.VideoWriter_fourcc(*format)  # define the video codec

    video = cv2.VideoWriter(output_video_path, fourcc, 1.0, size, is_color)

    for image in images:
        for _ in range(each_image_duration):
            cv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

            video.write(cv_image)

    video.release()

    cv2.destroyAllWindows()
