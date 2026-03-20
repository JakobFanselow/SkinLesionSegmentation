#import "@preview/bloated-neurips:0.7.0": botrule, midrule, neurips2025, paragraph, toprule, url
#import "logo.typ": LaTeX, LaTeXe, TeX

#let affls = (
  hpi: (
    department: "Student",
    institution: "Hasso-Plattner-Institut",
    location: "Potsdam",
  )
)

#let authors = (
  (name: "Jakob Fanselow",
   affl: "hpi",
   email: "jakob.fanselow@student.hpi.de",
   equal: true),
  (name: "Tim Beier", 
   affl: ("hpi"), 
   email: "tim.beier@student.hpi.de",
   equal: true),
)

#let line = box(
  width: 100%,
  height: 0.2pt,
  fill: black
)

#show: neurips2025.with(
  title: [Skin Lesion Segmentation using Deep Neural Networks: A Comparison of Different Approaches on the ISIC 2018 Dataset],
  authors: (authors, affls),
  keywords: ("Machine Learning", "NeurIPS"),
  abstract: [
    // When writing the summary, always explain the ‘why’ after you specified the ‘what’.
  ],
  bibliography: bibliography("main.bib"),
  appendix: [],
  accepted: true,
)

// IMPORTANT: This file should be exported in Times New Roman. On the HPC-Cluster I (Tim) wasn't able to get the template to recognize Times New Roman so I would have to export this outside the cluster for it to use Times New Roman.

= Introduction
// What is your project about? What are its goals?

Medical image segmentation plays an important role in the analysis of medical images, often used for isolating regions of interest from their background @medimagereview. This assists trained personal in in identifying lesions, their location, size and potentially relationship with surrounding tissues. Compared to natural images, skin lesion images come with unique challenges like low contrast with surrounding skin, variations in lesion appearance, presence of artifacts like hair and illumination changes. \ \ 
In this project, we compare the performance of multiple deep learning architectures for skin lesion segmentation. Specifically, we compare three models: U-Net, V-Net and DeepLabV3. While U-Net and V-Net are based on the classic encoder-decoder architecture, DeepLabV3 makes use of atrous spatial pyramid pooling (ASPP). \ \  
#line

Skin cancer is one of the most common forms of cancer worldwide, with melanoma being the most aggressive and life-threatening variant [1]. Early detection is critical, as the prognosis strongly depends on identifying malignant lesions at an early stage [2]. In clinical practice, dermatologists rely on visual examination and dermoscopic imaging to assess suspicious skin lesions; however, manual analysis is time-consuming and subject to inter-observer variability [3].

Within automated diagnostic systems, image segmentation plays a central role. Skin lesion segmentation aims to accurately delineate the boundary between the lesion and the surrounding healthy skin. This step is crucial because it directly affects subsequent tasks such as feature extraction and classification [4]. Precise segmentation enables the extraction of meaningful characteristics, including shape, color, and texture, which are essential for distinguishing between benign and malignant lesions [5].

Despite significant advances in deep learning, skin lesion segmentation remains a challenging problem. Dermoscopic images exhibit high variability in lesion appearance, including differences in size, shape, and color [6]. In many cases, the contrast between the lesion and surrounding skin is low, making boundary detection difficult [7]. Additionally, images often contain artifacts such as hair, air bubbles, or illumination variations that can obscure relevant structures [6]. Another major challenge is the limited availability of annotated medical data, as pixel-wise labeling requires expert knowledge and is time-consuming [8]. Furthermore, class imbalance—where lesion regions occupy only a small portion of the image—can negatively impact model training [9].


= Related Work
// Have others approached what you did? Which works are related to yours?
A substantial body of research has explored deep learning approaches for skin lesion segmentation, with particular focus on encoder–decoder architectures and multi-scale convolutional models. Among these, U-Net has become one of the most widely adopted architectures in medical image segmentation due to its ability to capture both local and global context through skip connections [1]. Numerous studies have demonstrated its strong performance on dermoscopic datasets, including ISIC challenges, often serving as a baseline for comparison [2].

V-Net, originally proposed for volumetric medical image segmentation, extends the encoder–decoder paradigm by incorporating residual connections and volumetric convolutions [3]. While primarily designed for 3D data, variants of V-Net have also been applied to 2D medical imaging tasks, including skin lesion segmentation, showing competitive performance in handling complex structures and class imbalance [4].

In parallel, architectures based on atrous (dilated) convolutions, such as DeepLabV3, have been successfully applied to segmentation tasks by enabling multi-scale context aggregation without reducing spatial resolution [5]. Although originally developed for natural image segmentation, DeepLab-based models have increasingly been adapted to medical imaging problems, including dermoscopic image analysis [6].

The ISIC 2018 dataset has emerged as a standard benchmark for evaluating segmentation methods in this domain [2]. Many studies report results on this dataset, often comparing variations of U-Net or hybrid architectures. However, direct and systematic comparisons between fundamentally different architectural paradigms—such as encoder–decoder models (U-Net, V-Net) and atrous convolution-based models (DeepLabV3)—remain relatively limited. In particular, few works investigate these models under consistent experimental conditions or analyze the impact of training configurations in a controlled setting.

In this work, we address this gap by providing a structured comparison of U-Net, V-Net, and DeepLabV3 on the ISIC 2018 dataset. In addition to evaluating their segmentation performance, we implement U-Net and V-Net from scratch in PyTorch and conduct a hyperparameter sweep for U-Net, enabling a more detailed analysis of how architectural and training choices influence performance.

= Dataset
// What are characteristics of your dataset, e.g. size, input/target output, dimensions, conducted
In this work, we use the ISIC 2018 Challenge dataset, which is a widely adopted benchmark for skin lesion analysis and segmentation tasks [1]. The dataset consists of dermoscopic images acquired under varying conditions, providing a realistic and challenging setting for evaluating segmentation models.

The segmentation task is formulated as a pixel-wise binary classification problem, where the input is a dermoscopic image and the target output is a corresponding binary mask indicating lesion versus non-lesion regions. Each image is annotated by experts, ensuring high-quality ground truth labels, although the annotation process is inherently time-consuming and subject to some variability.

The dataset contains 2,594 training images, each paired with a segmentation mask. The images vary in spatial resolution, typically around 600 × 450 pixels, though exact dimensions are not fixed and differ across samples. This variability reflects real-world acquisition conditions but requires preprocessing steps such as resizing and normalization to enable efficient model training.

A key characteristic of the dataset is the high variability in lesion appearance, including differences in size, shape, color, and texture. Additionally, many images contain artifacts such as hair, shadows, or illumination inconsistencies, which further complicate the segmentation task. Another important aspect is the class imbalance between foreground (lesion) and background pixels, as lesions often occupy only a small portion of the image.

To ensure consistent training and evaluation, the dataset is typically split into training, validation, and test subsets. In our experiments, we follow a standard split of the ISIC 2018 dataset, using the provided training set and reserving a portion for validation. All images are preprocessed by resizing them to a fixed resolution and applying normalization, while data augmentation techniques such as flipping, rotation, and scaling are employed to improve generalization.

Overall, the ISIC 2018 dataset provides a diverse and challenging benchmark for skin lesion segmentation, making it well-suited for comparing different deep learning architectures under realistic conditions.

= Methods
// Which machine learning architecture have you chosen? How have you trained your model? Which experiments have you run? How did you select hyperparameters?preprocessing, dataset splits

== U-Net
U-Net is a convolutional encoder–decoder architecture specifically designed for biomedical image segmentation [1]. It consists of a contracting path (encoder) that captures context through successive convolution and pooling operations, and an expanding path (decoder) that restores spatial resolution via upsampling. A key feature of U-Net is the use of skip connections, which transfer high-resolution feature maps from the encoder to the corresponding decoder layers. This allows the model to combine semantic information with fine-grained spatial details, leading to precise localization.

In the context of skin lesion segmentation, U-Net is particularly effective because it can preserve boundary information while learning global structures. However, its performance can be sensitive to hyperparameters and training setup, and it may struggle when lesions exhibit extreme variability or when contextual information beyond local neighborhoods is required.

== V-Net
V-Net extends the encoder–decoder paradigm by incorporating residual connections and was originally designed for volumetric (3D) medical image segmentation [2]. Instead of simple skip connections as in U-Net, V-Net uses residual blocks within each stage, which facilitate gradient flow and allow for deeper architectures. 

When adapted to 2D tasks such as skin lesion segmentation, V-Net maintains its core design principles. Compared to U-Net, the residual connections can improve training stability and enable better feature reuse, potentially leading to improved performance on complex structures. However, the architecture is typically more computationally intensive, and its benefits over U-Net are not always consistent in 2D settings.

== DeepLabV3
DeepLabV3 departs from the classical encoder–decoder structure and instead focuses on atrous (dilated) convolutions to capture multi-scale contextual information [3]. By using dilated convolutions with different rates, DeepLabV3 can increase the receptive field without reducing spatial resolution. This is complemented by the Atrous Spatial Pyramid Pooling (ASPP) module, which aggregates features at multiple scales.

In contrast to U-Net and V-Net, DeepLabV3 does not rely on symmetric skip connections for precise localization. Instead, it emphasizes global context, which can be advantageous when segmenting objects with large variability in size and appearance. However, the lack of strong low-level feature fusion may result in less precise boundaries, particularly in medical images where fine details are critical.


= Evaluation
// Which evaluation strategy and metrics have you chosen? How does your model perform?

= Discussion
// Is your model performing well? How could your model be improved? Which challenges were involved? What is future work?