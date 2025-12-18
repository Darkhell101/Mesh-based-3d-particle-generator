# Mesh-based-3d-particle-generator
Here is a 3D mesh-based particle generator for meshless method. Only a standalone python script.

![滑坡](https://github.com/Darkhell101/Mesh-based-3d-particle-generator/blob/main/model-generated.png)

Main file:
  particle-generator.py

Input:
  3D mesh (only surface mesh) or internal geometric objects

Output:
  Particles surrounded by mesh, particle volume, particle radius, packing type

Note:
  With the help of grok and qwen, various common packing types have achieved. Fcc is the best, complete isotropy, but more particles than scc.

  More information: https://www.gan.msm.cam.ac.uk/resources/crystalmodels/cubic

  ![晶体模型](https://github.com/Darkhell101/Mesh-based-3d-particle-generator/blob/main/model-generated.png)

Limitations:
  Accidents (Isolated point) may occur if there are grid points inside.

Acknowledgements:
  partial idea: https://opencax.cn/
  model data: Tao Pan, Rhino，CQU MD
