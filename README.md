# Master Thesis Degree code on Domain Adaptation for Semantic Segmentation between syntetic and real data

Autonomous driving is without a shadow of doubt the future of automotive and it currently needs efficient solutions. Deep learning methods have shown excellent results for object classification and detection, but the specific task needed for driving is the more challenging semantic segmentation: each pixel of a depicted scene should be recognized as belonging to a specific object. For this setting, it is crucial to collect a large amount of per-pixel labeled data, which need an expensive manual classification process. Synthetic datasets from simulators can be used in order to reduce the amount of data required and they come with free annotation by design. However, the big style difference between synthetic and real images does not allow a direct knowledge transfer across the two domains and it asks for specific domain adaptation solutions. This work starts from a review of the most recent literature on deep learning for semantic segmentation and it proposes to boost domain adaptation techniques based on Generative Adversarial Networks (GANs) by combining them with features-based adaptive strategies that take into consideration category-level adaptation and self-supervision. The experimental analysis considers a model learned on synthetic labeled images from the GTA V videogame and applied on the real Cityscapes dataset collected in different cities in Germany. The obtained results show how the proposed combination can be useful in case of limited availability of real world annotated images.

Thesis link: [link](https://webthesis.biblio.polito.it/14148/)


## IMAGE-TO-IMAGE TRANSLATION

- CycleGAN and Pix2Pix
- NVIDIA FastPhotoStyle
- SG-GAN
 
## DOMAIN ADAPTATION NETWORKS

- AdaptSegNet 
- CLAN
- BIDIRECTIONAL LEARNING (BDL)

## MY DOMAIN ADAPTATION NETWORK 
- Self-CLAN
