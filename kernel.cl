__kernel void quantify(__read_only image2d_t imgIn, __write_only image2d_t imgOut)
{
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;
  float4 q_colors[10] = {
                          (float4)(0.0f,0.0f,0.0f,0.0f),
                          (float4)(127.0f,0.0f,0.0f,0.0f),
                          (float4)(255.0f,0.0f,0.0f,0.0f),
                          (float4)(0.0f,127.0f,0.0f,0.0f),
                          (float4)(0.0f,255.0f,0.0f,0.0f),
                          (float4)(0.0f,0.0f,127.0f,0.0f),
                          (float4)(0.0f,0.0f,255.0f,0.0f),
                          (float4)(127.0f,0.0f,127.0f,0.0f),
                          (float4)(127.0f,127.0f,0.0f,0.0f),
                          (float4)(0.0f,127.0f,127.0f,0.0f)
                        };
  int q_levels = 10;
  float q_step = 255.0f / q_levels;
  int2 currentPosition = (int2)(get_global_id(0), get_global_id(1));
  float4 currentPixel = 0.0f;
  float4 calculatedPixel = 0.0f;
  currentPixel = read_imagef(imgIn, smp, currentPosition);
  float intensity = (currentPixel.s0 + currentPixel.s1 + currentPixel.s2) / 3.0f;
  int level = 0;

  for (int l = 0; l < q_levels;)
    if (intensity < (l + 1) * q_step) {
      level = l;
      break;
    }

  calculatedPixel = q_colors[level];
  write_imagef(imgOut, currentPosition, calculatedPixel);
}