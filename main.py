import sys

import pyopencl as cl
import numpy as np
import time
from PIL import Image
import cv2
import os

q_level = 10
q_step = 255 / q_level
q_colors = [(0, 0, 0), (127, 0, 0), (255, 0, 0), (0, 127, 0), (0, 255, 0), (0, 0, 127), (0, 0, 255), (127, 0, 127),
            (127, 127, 0), (0, 127, 127)]


def level(red, green, blue):
    intensity = (red + green + blue) / 3

    for lev in range(10):
        if intensity < (lev + 1) * q_step:
            return lev

    return 0


# Image processing without GPU
print('Image processing without GPU')

for k in range(10):
    print('Iteration %s' % k)
    for n in range(3):
        pil_img = Image.open('img/%s.png' % n)
        img = pil_img.convert('RGB')
        start = time.time()

        for i in range(img.size[0]):
            for j in range(img.size[1]):
                r, g, b = img.getpixel((i, j))
                img.putpixel((i, j), q_colors[level(r, g, b)])

        img.save('img/%s_CPU.png' % n)
        end = time.time() - start
        print('№%s: %s seconds' % (n + 1, round(end, 2)))

# Image processing with GPU
print('Image processing with GPU')
platforms = cl.get_platforms()
# AMD Accelerated Parallel Processing
platform = platforms[0]
devices = platform.get_devices(device_type=cl.device_type.GPU)
# gfx902
device = devices[0]
context = cl.Context([device])
cQ = cl.CommandQueue(context, device)

for k in range(0):
    print('Iteration %s' % k)
    for n in range(3):
        imgIn = cv2.imread('img/%s.png' % n, cv2.COLOR_BGR2RGB)
        # imgIn = np.asarray(imgIn)
        imgOut = np.zeros_like(imgIn)
        # imgInBuf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imgIn)
        # imgOutBuf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imgOut.nbytes)
        start = time.time()
        shape = imgIn.T.shape
        # [1::-1]
        imgInBuf = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8), shape=shape)
        imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8), shape=shape)
        with open('kernel.cl') as code:
            program = cl.Program(context, code.read()).build()
        kernel = cl.Kernel(program, 'quantify')
        kernel.set_arg(0, imgInBuf)
        kernel.set_arg(1, imgOutBuf)
        # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        cl.enqueue_nd_range_kernel(cQ, kernel, shape, None)
        cl.enqueue_copy(cQ, imgInBuf, imgIn, origin=(0, 0), region=shape, is_blocking=False)
        cl.enqueue_nd_range_kernel(cQ, kernel, shape, None)
        # program.quantify(cQ, imgIn.shape, None, imgInBuf, imgOutBuf).wait()
        cl.enqueue_copy(cQ, imgOut, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True)
        # cl.enqueue_copy(cQ, imgOut, imgOutBuf).wait()
        cv2.imwrite(os.path.join('img/%s_CL.png' % n), imgOut)
        end = time.time() - start
        print('№%s: %s seconds' % (n + 1, round(end, 2)))
