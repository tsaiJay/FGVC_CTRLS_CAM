# FGVC_CTRLS_CAM
paper url : https://ieeexplore.ieee.org/document/10444242

In this paper, we focus on knowledge distillation specifically designed for fine-grained image recognition. Notably, this strategy is inspired by the Class Activation Maps (CAM). We first train a complex model and use it generated feature maps with spatial information, which is call hint maps. Furthermore, we propose an adjustment strategy for this hint map, which can control local distribution of information. Referring to it as Controllable Size CAM (CTRLS-CAM). Use it as a guide allowing the student model to effectively learn from the teacher model’s behavior and concentrate on discriminate details. In contrast to conventional distillation models, this strategy proves particularly advantageous for fine-grained recognition, enhancing learning outcomes and enabling the student model to achieve superior performance. In CTRLS-CAM method, we refine the hint maps value distribution, redefining the relative relationships between primary and secondary feature areas. We conducted experiments using the CUB200-2011 dataset, and our results demonstrated a significant accuracy improvement of about 5% compared to the original non-distilled student model. Moreover, our approach achieved a 1.23% enhancement over traditional knowledge distillation methods.

![framework](./img/structure.png)

in this implementation
also ditillation on other datasets 

accuracy table
!!!!!!!!!!

1. Data Preparation

2. Training Teacher & Student model

3. Distillation
   ```
   python main.py --c cfg_setting.yaml
   ```
