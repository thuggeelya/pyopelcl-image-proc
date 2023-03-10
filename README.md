# Image processing on GPU by using PyOpencl framework
- Estimation of the program performance using the average program execution time for at least 10 launches.
- Compare the performance for various image sizes: 1024Ρ768, 1280Ρ960, 2048Ρ1536.
- Compare the performance of the program on both CPU and GPU (OpenCL).
# Image quantization
Get intensity values πΌ = (πππ + πΊππππ + π΅ππ’π)/3,
where πΌ β pixel intensity, πππ β red component value, πΊππππ β green component, π΅ππ’πβ blue component.
Set the number of quantization levels K - any integer in the range from 4 to 10 and split the intensity scale
into quantization levels. For quants, match the following colors:
- 1st quantum RGB=(0,0,0)
- 2nd quantum RGB=(127,0,0)
- 3rd quantum RGB=(255,0,0)
- 4th quantum RGB=(0,127,0)
- 5th quantum RGB=(0,255,0)
- 6th quantum RGB=(0,0,127)
- 7th quantum RGB=(0,0,255)
- 8th quantum RGB=(127,0,127)
- 9th quantum RGB=(127,127,0)
- 10th quantum RGB=(0,127,127)

For each pixel, replace the color channel values with the color values of the quantum to which the intensity
of this pixel belongs. Save the resulting image to a file.
