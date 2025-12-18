# Mesh-Based 3D Particle Generator

A standalone Python script for generating 3D particles inside a surface mesh, designed for meshless (particle-based) numerical methods.

![Landslide Model](https://github.com/Darkhell101/Mesh-based-3d-particle-generator/blob/main/model-generated.png)

## 📄 Main File
- `particle-generator.py`

## 📥 Input
- A **3D surface mesh** (e.g., STL, OBJ) **or**
- Built-in geometric primitives (e.g., box, sphere)

## 📤 Output
- Particle coordinates inside the enclosed volume  
- Per-particle **radius** and **volume**  
- Selected **packing type** (SC / BCC / FCC)

## 🔬 Packing Types

With help from Grok and Qwen, common cubic crystal packings have been implemented:

| Type | Full Name               | Features                                      |
|------|-------------------------|-----------------------------------------------|
| SC   | Simple Cubic            | Low density, anisotropic, fewest particles     |
| BCC  | Body-Centered Cubic     | Moderate density and isotropy                 |
| FCC  | Face-Centered Cubic     | **Highest packing density**, fully isotropic — preferred for accuracy (but uses more particles) |

> 💡 Learn more about cubic crystal structures:  
> [Common Cubic Structures – University of Cambridge](https://www.gan.msm.cam.ac.uk/resources/crystalmodels/cubic)

![Crystal Packing Models](https://github.com/Darkhell101/Mesh-based-3d-particle-generator/blob/main/crystal.png)

## ⚠️ Limitations
- **Isolated particles** may appear if the mesh contains internal vertices or self-intersections.  
- Ensure the input mesh is **watertight** and **manifold** for reliable results.

## 🙏 Acknowledgements
- Conceptual inspiration: [OpenCAX](https://opencax.cn/)  
- Model data & visualization: **Tao Pan** (Rhino, CQU MD)
