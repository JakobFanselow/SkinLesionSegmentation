This project implements and bechnmarks two state-of-the-art models for image segmentation. Namely U-Net and a 2D adaption of V-Net

Detailed results and instructions to replicate our results can be found in our [demonstration notebook](demo.ipynb).

Our UNet implementation is located [here](unet/unet.py) and our V-Net adaption can be found [here](vnet/vnet.py).


*Dataset*
ISIC 2018 Challenge Task 1: Lesion segmentation: Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368

*U-Net Architecture*
O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation.” [Online]. Available: https://arxiv.org/abs/1505.04597

*V-Net Architecture*
F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.” [Online]. Available: https://arxiv.org/abs/1606.047979
