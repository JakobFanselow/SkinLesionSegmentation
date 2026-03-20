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

Skin cancer is one of the most common forms of cancer worldwide, with melanoma being the most aggressive and life-threatening variant of skin cancer @heistein2024melanoma. Early detection is critical, as the life expectancy strongly correlates with identifying malignant lesions at an early point in time @heistein2024melanoma. Dermatologists rely on visual examination and dermoscopic imaging to find suspicious skin lesions. This manual analysis is time-consuming and subject to inter-observer variability @demoscopyofpigmentedskinlesions.

Within automated melanoma diagnostic systems, image segmentation plays a central role. Skin lesion segmentation aims to accurately delineate the boundary between the lesion and the surrounding healthy skin. This step is crucial because it directly affects subsequent tasks such as feature extraction and classification @codella2019skinlesionanalysismelanoma.

Despite significant advances in deep learning, skin lesion segmentation remains a challenging problem. This has several reasons: Dermoscopic images exhibit high variability in lesion appearance, including differences in size, shape, and color @mutualbootstrapping2020. In many cases, the contrast between the lesion and surrounding skin is low, making boundary detection difficult @automaticskinlesionsegmentation. Additionally, images often contain artifacts such as hair, air bubbles, or illumination variations that can obscure relevant structures @mutualbootstrapping2020. Another major challenge is the limited availability of annotated medical data, as pixel-wise labeling requires expert knowledge and is time-consuming @LITJENS201760. These factors contribute to the complexity of skin lesion segmentation and motivate the need for robust and effective deep learning models that can handle such variability and challenges.


= Related Work
// Have others approached what you did? Which works are related to yours?
Lots of research has explored use cases of deep learning for skin lesion segmentation, with focus on encoder–decoder architectures. U-Net was one of the most widely adopted encoder-decoder architectures in medical image segmentation due to its ability to capture both local and global context through the introduction of skip connections @ronneberger2015unetconvolutionalnetworksbiomedical. Many studies demonstrated its strong performance on dermoscopic datasets, including ISIC challenges, such as ISIC 2018, often serving as a baseline for comparison @codella2019skinlesionanalysismelanoma.

V-Net, which was originally proposed for volumetric medical image segmentation, extends the encoder–decoder paradigm by incorporating residual connections and volumetric convolutions @milletari2016vnetfullyconvolutionalneural. While primarily designed for 3D data, variants of V-Net have also been applied to 2D medical imaging tasks, including skin lesion segmentation, showing competitive performance in handling complex structures and class imbalance @Hashemi_2019.

I our project, we provide a structured comparison of U-Net and V-Net on the ISIC 2018 dataset. In addition to evaluating their segmentation performance, we implement U-Net and V-Net from scratch in PyTorch and conduct a hyperparameter sweep, enabling a more detailed analysis of how architectural and training choices influence performance.

= Dataset
// What are characteristics of your dataset, e.g. size, input/target output, dimensions, conducted
In this work, we use the ISIC 2018 Challenge dataset, which is a widely adopted benchmark for skin lesion analysis and segmentation tasks @codella2019skinlesionanalysismelanoma. The dataset consists of dermoscopic images acquired under varying conditions and institutions, providing a realistic and challenging setting for evaluating segmentation models.

The segmentation task is formulated as a pixel-wise binary classification problem, where the input is a dermoscopic image and the target output is a corresponding binary mask indicating lesion versus non-lesion regions. Each image is annotated by experts, ensuring high-quality ground truth labels, although the annotation process is inherently time-consuming and subject to some variability as seen in @fig1

#v(1em)
#figure(
  grid(
    columns: 4,
    column-gutter: 1em,
    row-gutter: 1em,
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input/ISIC_0000003.jpg"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth/ISIC_0000003_segmentation.png"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input/ISIC_0000024.jpg", alt: "Sample Image 2"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth/ISIC_0000024_segmentation.png", alt: "Sample Image 2 Mask"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input/ISIC_0000071.jpg", alt: "Sample Image 3"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth/ISIC_0000071_segmentation.png", alt: "Sample Image 3 Mask"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1-2_Training_Input/ISIC_0016055.jpg", alt: "Sample Image 4"),
    image("../isic2018-challenge-task1-data-segmentation/ISIC2018_Task1_Training_GroundTruth/ISIC_0016055_segmentation.png", alt: "Sample Image 4 Mask"),
  ),
  caption: [Example images and corresponding segmentation masks from the ISIC 2018 dataset.]
) <fig1>
#v(1em)

The dataset consists of 2,594 training images, with each having a corresponding segmentation mask. One characteristic of the dataset is the substantial variability in lesion appearances (@fig1), encompassing differences in size, shape, color, and texture, and also labeling quality. Numerous images contain artifacts such as hair, shadows, or inconsistent illumination, which increase the complexity of the segmentation task. Another challenge is the class imbalance between foreground (lesion) and background pixels, as lesions frequently take up only a small fraction of the image.

For training and evaluation, the dataset is divided into training, validation, and test subsets. For this we split the ISIC 2018 training dataset using a 70-15-15 ration. To achieve a larger training datset rotations and mirroring are applied. Furthermore all images undergo preprocessing by resizing to a fixed resolution ($512 times 512$).

A limitation of the ISIC 2018 dataset is its limited representation of the diversity of skin lesions encountered in clinical practice, as it primarily comprises dermoscopic images from specific sources e.g. primarly people with lighter skin tones. Additionally, the dataset only labels a binary segmentation mask, which does not encompass the full spectrum of clinical scenarios where multi-class segmentation, e.g. differentiating between various lesion types, may be necessary. Even with these limitations, the ISIC 2018 dataset remains a valuable resource for benchmarking segmentation models.

= Methods
// Which machine learning architecture have you chosen? How have you trained your model? Which experiments have you run? How did you select hyperparameters?preprocessing, dataset splits

== U-Net
U-Net is an encoder–decoder architecture specifically designed for biomedical image segmentation @unet. It was the first to introduce skip connections in the context of image segmentation. The U-Net architecture contains an encoder, first half, which captures context through convolution and pooling operations, and a decoder, second half, which restores resolution via upsampling and skip connections. These skip connections transfer high-resolution feature maps from encoder to decoder, which allows the model to combine global semantic information with local spatial details, which leads to precise segmentation.
#v(1em)
#figure(image("/assets/image.png",width: 70%),
caption: [U-Net architecture (adapted from @unet)])<fig2>
#v(1em)

In each stage the encoder applies two consecutive $3 times 3$ convolutions, followed by an activation function, typically ReLU (see @fig2). A $2 times 2$ max-pooling operation is used for downsampling and with each step, the number of feature channels is doubled, which enables U-Net to learn increasingly abstract representations. At the bottleneck, the network captures high-level features with a large receptive field.

In the decoder, spatial resolution is restored using upsampling operations (transposed convolutions). After each step, the corresponding feature map from the encoder is concatenated via skip connections. The resulting feature map is processed by convolutional layers, which enables the model to find spatial details using both local and contextual information. At last, a $1 times 1$ convolution maps the feature representation to the desired number of output classes.

The original U-Net architecture employs valid convolutions, which reduce spatial dimensions after each convolution ($572 times 572 -> 570 times 570$). Consequently, encoder feature maps must be cropped before concatenation with decoder features. To address this, many modern U-Net implementations use padding to maintain spatial dimensions, which simplifies the architecture and allows for more flexible input sizes, as will be adopted in this work.

For skin lesion segmentation, U-Net is particularly effective because it preserves boundary information while capturing global structures. However, its performance is sensitive to hyperparameter selection and training configuration, and it may struggle when lesions exhibit extreme variability or when contextual information beyond local neighborhoods is necessary.

== V-Net
V-Net extends the encoder–decoder paradigm by incorporating residual connections and was originally designed for volumetric (3D) medical image segmentation @vnet. Similar to U-Net, it follows a contracting–expanding structure, but replaces standard convolutional blocks with residual blocks, enabling more efficient gradient propagation and facilitating the training of deeper networks.
#v(1em)
#figure(
  image("/assets/image-1.png", width: 70%),
  caption: [V-Net architecture (adapted from @vnet)]
)<fig3>
#v(1em)
In the original 3D formulation, V-Net operates on volumetric inputs using 3D convolutions (e.g., $5 times 5 times 5$) and 3D pooling/upsampling operations (see @fig3). Each stage consists of one or more convolutional layers combined with a residual connection, where the input of a block is added to its output. Downsampling is typically performed using strided convolutions rather than max pooling, while upsampling is achieved via transposed convolutions. Similar to U-Net, skip connections between encoder and decoder stages are also used, but the internal structure of each stage is more complex due to the residual design.

To adapt V-Net to 2D tasks such as skin lesion segmentation, the architecture is modified by replacing all 3D operations with their 2D counterparts. Specifically:
-	$5 times 5 times 5$ convolutions → $5 times 5$ convolutions
-	3D feature maps → 2D feature maps
-	volumetric input → standard RGB images
-	3D up/downsampling → 2D strided or transposed convolutions

This transformation preserves the overall architectural design while making the model applicable to planar image data. However, it also removes the ability to exploit inter-slice spatial context, which is a key advantage of the original 3D formulation in medical imaging tasks such as MRI or CT segmentation.

Compared to U-Net, V-Net introduces residual learning within each resolution level, which can improve convergence behavior and robustness, particularly in deeper networks. @vnet

In the context of skin lesion segmentation, the 2D adaptation of V-Net benefits from improved feature reuse and potentially more stable training due to residual connections. However, the increased architectural complexity leads to higher computational cost, and the absence of true volumetric context might limit its advantages over simpler architectures like U-Net.

== Optimization strategy
To improve training we implemented the following training heuristics:
+ We utilized the AdamW which decouples weight decay from the gradient update, resulting in better generalization than Adam
+ The OneCycleLR learning rate scheduler which starts with a low but increasing learning rate until it hits a maximum after which the learning rate decays
+ To ensure training stability we implemented gradient clipping with a max_norm hyperparameter. This serves as a safety mechanism against "exploding gradients" by scaling down gradient that exceed the max_norm threshhold
+ Data augmentation: In medical imaging of skin lesions the orientation of an image or whether it is flipped or not doesn't matter. We randomly applied rotations by multiples of 90° and mirroring which effectively allowed us to increase our training data by a factor of 8.

== Sweeps <Sweeps>
To ensure we are comparing the optimal versions of our models we used wandb sweeps with the bayes method. The hyper parameters of which we explored variations are:
+ LR
+ max LR 
+ max norm
+ batch size
+ weight decay
+ size of the kernel used in convolutions (with padding added to ensure matching dimensions)
+ number of epochs

To avoid asymetry issues we refrained from using even kernel sizes.
Additionally in our UNet sweep, we also experimented with removing the bottleneck connection.
The ranges which the sweep explored are:
#table(
  columns: (auto, 1fr, 1fr),
  inset: 10pt,
  align: horizon,
  fill: (x, y) => if y == 0 { gray.lighten(80%) },
  [*Parameter*], [*Distribution*], [*Min / Max / Values*],
  [learning_rate], [log_uniform_values], [$1e-6$ to $1e-3$],
  [max_learning_rate], [log_uniform_values], [$1e-5$ to $1e-2$],
  [max_norm], [categorical], [0.1, 0.25, 0.5, 1.0, 2.0],
  [batch_size], [categorical], [2, 4, 8, 16],
  [weight_decay], [log_uniform_values], [$1e-8$ to $0.5$],
  [kernel_size], [categorical], [3, 5, 7, 9],
  [epochs], [int_uniform], [15 to 40],
)


= Evaluation
// Which evaluation strategy and metrics have you chosen? How does your model perform?
== Strategy and Metric
We evaluated two state-of-the-art segmentation models.
The models were trained using the hybrid loss function Dice-BCE Loss which combines spatial overlap benefits of the Dice coefficient with strong pixel-wise classification of Binary Cross-Entropy
To ensure the architectural and model hyperparameter comparison remained the main focus, we applied an equal weighting to both components: 
$ L_"total" = 0.5 dot L_"Dice" + 0.5 dot L_"BCE" $

This provides a standardized baseline, allowing for an evaluation of architecture and model hyperparameters.

For hyperparameter choice see @Sweeps Sweeps.

== Quantitative Results
Using the best parameters found in our sweeps we were able to achieve the following results:
#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: center + horizon,
    stroke: none,
    table.hline(),
    [*Model*], [*Train loss*], [*Validation Loss*], [*Test Loss*], [*Dice coefficient*],
    table.hline(stroke: .5pt),
    
    [UNet], [*0.11005*], [*0.14995*], [*0.15901*], [*0.85149*],
    [VNet], [0.12168], [0.15772], [0.16998], [*0.85109*],
    
    table.hline(),
  ),
  caption: [Performance comparison of UNet and VNet using Dice-BCE loss.],
)

While UNet was able to achieve a 6.453% lower loss than VNet, their dice coefficients are nearly identical, from which follows that the lower loss of VNet is solely caused by UNet performing better in terms of BCE loss.
(Expand on UNet not beating VNet in Dice)

To visualize the difference, we randomly selected four random samples from the test set for qualitative review.
#image("../results/1.png")
#image("../results/2.png")
#image("../results/3.png")
#image("../results/4.png")

The visual evidence supports our quantitative findings. While both are able to successfully identify the lesions, VNet exhibits higher "uncertainty". This manifests in faint activations in areas with healthy skin. This is reflected in UNet outperforming VNet in terms of BCE loss.



= Discussion
// Is your model performing well? How could your model be improved? Which challenges were involved? What is future work?
==
A notable divergence in training was observed in stability and gradient size. While UNet remained stable under standard conditions, VNet was only able to perform well with very restrictive gradient clipping. Comparing the max_norm of the two best runs UNet had relatively stable training using a value of 2.0 while VNet's loss curves exhibit large spikes even though its max_norm was only 0.25

#image("../training_curves/TrainLoss.png")
#image("../training_curves/ValLoss.png")

This restrictive max_norm resulted in a clipping of almost all of VNets gradients.

#image("../training_curves/GradientClipping.png")
 
The qualitative "uncertainty" may be a direct consequence of this restriction. Because of the low max_norm required to keep the training process stable, VNet was restricted in its ability to perform high-magnitude weight updates.
Noteably it was also unable to take advantage of our learning rate scheduler as nearly all gradients were clipped to the same size.
This likely prevented VNet from reaching the same level of confidence in pixel-wise classification (low BCE loss) as UNet.
However it has to be noted that because the best UNet run had a lower max_lr than lr our training program used $1.5*"lr"$ as a fallback max_lr, limiting Unet's advantage.

== 
Despite UNets superior BCE performance, the parity of dice coefficients suggests that both architectures are equally capable of capturing the primary structural morphology of the lesion.

== Future work
=== Architectural Stabilization
The instability in VNet's training characterized by large loss spikes and the neccecity for aggressive gradient clipping, remains a primary bottleneck. Future work should investigate the following techniques:
+ initialization besides random initialization such as He initialization
+ residual scaling
=== Loss Function Optimization
To address VNets poor BCE performance we propose implementing a scheduled loss weighting strategy that initially prioritizes Dice and later on BCE loss.