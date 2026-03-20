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

= Related Work
// Have others approached what you did? Which works are related to yours?


= Dataset
// What are characteristics of your dataset, e.g. size, input/target output, dimensions, conducted

= Methods
// Which machine learning architecture have you chosen? How have you trained your model? Which experiments have you run? How did you select hyperparameters?preprocessing, dataset splits

= Evaluation
// Which evaluation strategy and metrics have you chosen? How does your model perform?

= Discussion
// Is your model performing well? How could your model be improved? Which challenges were involved? What is future work?