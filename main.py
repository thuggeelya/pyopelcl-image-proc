import pyopencl as cl
import numpy as np
import cv2
import os

platforms = cl.get_platforms()
platform = platforms[0]
devices = platform.get_devices(cl.device_type.GPU)
device = devices[0]
context = cl.Context([device])
cQ = cl.CommandQueue(context, device)
kernel_code = """
__kernel void quantify(__read_only image2d_t imgIn, __write_only image2d_t imgOut)
{
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  float4 q_colors[10] = {
                          (float4)(0,0,0,0),
                          (float4)(127,0,0,0),
                          (float4)(255,0,0,0),
                          (float4)(0,127,0,0),
                          (float4)(0,255,0,0),
                          (float4)(0,0,127,0),
                          (float4)(0,0,255,0),
                          (float4)(127,0,127,0),
                          (float4)(127,127,0,0),
                          (float4)(0,127,127,0)
                        };
  int q_levels = 10;
  float q_step = 255/q_levels;
  int2 currentPosition = (int2)(get_global_id(0), get_global_id(1));
  float4 currentPixel = (float4)(0,0,0,0);   
  float4 calculatedPixel = (float4)(0,0,0,0);
  currentPixel = read_imagef(imgIn, smp, currentPosition);
  int level = 0;
  float r = currentPixel.x;
  float g = currentPixel.y;
  float b = currentPixel.z;
  float intensity = (r + g + b) / 3;

  for (int l = 0; l < q_levels;) {
    if (intensity < (l + 1) * q_step) {
      level = l;
      break;
    }
  }

  calculatedPixel = q_colors[level];
  write_imagef(imgOut, currentPosition, calculatedPixel);
}
"""

imgIn = cv2.imread("img/" + str(0) + ".png", cv2.IMREAD_UNCHANGED)
imgOut = np.empty_like(imgIn)
# imgInBuf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = imgIn)
# imgOutBuf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, imgOut.nbytes)
program = cl.Program(context, kernel_code).build()

shape = imgIn.T.shape
imgInBuf = cl.Image(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    cl.ImageFormat(cl.channel_order.RGBA,
                                   cl.channel_type.UNORM_INT8),
                    shape=shape)
imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY,
                     cl.ImageFormat(cl.channel_order.RGBA,
                                    cl.channel_type.UNORM_INT8),
                     shape=shape)
kernel = cl.Kernel(program, 'quantify')
kernel.set_arg(0, imgInBuf)
kernel.set_arg(1, imgOutBuf)
cl.enqueue_copy(cQ, imgInBuf, imgIn, origin=(0, 0), region=shape, is_blocking=False)
cl.enqueue_nd_range_kernel(cQ, kernel, shape, None)
cl.enqueue_copy(cQ, imgOut, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True)

cv2.imwrite(os.path.join('/img/' + str(0), '.png'), imgOut)
cv2.waitKey(0)
