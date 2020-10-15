# CS2952G-Team-4 (Blue Genes)
The final project repository for team 4 (Blue Genes) of CS2952G .


## Abstract
In recent years, Hi-C experiments are widely used to analyze chromatin interactions. But many applications regarding using Hi-C data are facing the problem that available Hi-C data have low resolution, which will hurt the related analysis. To address this problem, deep learning models such as a convolutional neural network (CNN) and a generative adversarial network (GAN) are used to enhance data resolution due to their effectiveness in various image processing tasks. The estimations of these models indeed increase the data resolution, however, the training cost is a significant increase as well. Previous papers pay little attention to compare the computational resources used in a different model. Moreover, these models consider the Hi-C data as images and apply correlation and image-based metrics to evaluate similarities between their estimations and high-resolution counterparts.  Therefore, the feasibility of these models in real applications is doubtful. In this paper, we implement comprehensive experiments to compare most of the ad hoc models on enhancing Hi-C resolution and utilize four biologically plausible similarity scores to measure the estimation. Based on the experimental results, we give a guidance on how to choose from various methods to best fit the application requirement and available computational resources. (If possible, maybe we can build up a new model.)   

## Project Goal
[a] Use four Hi-C-based measurements: GenomeDISCO, Hi-C Spector, HiCRep, and QuASAR-Rep. These metrics are proposed three years ago. So it would be better if other newly proposed measurements can be used.   

[b] Compare various deep learning models. Use the above-mentioned metrics. Compare their performances on various downstream analyses. Compares their required computational resources.

[c] Give guidance about how to choose a model for a Hi-C analysis application.

## Paper List

I think the literatur ecan be divided into three parts: What about each of us work on one part.

### GAN and its variations
[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

[2] Wang, Xintao, et al. "Esrgan: Enhanced super-resolution generative adversarial networks." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

**Paper [2] has a detailed literature review in the "Related Work" section. I recommend you refer to it for more knowledge about super-resolution GAN.**

### Deep learning models used for enhancing the Hi-C resolution
[3] Li, Zhilan, and Zhiming Dai. "SRHiC: A Deep Learning Model to Enhance the Resolution of Hi-C Data." Frontiers in Genetics 11 (2020): 353.

[4] Liu, Tong, and Zheng Wang. "HiCNN: a very deep convolutional neural network to better enhance the resolution of Hi-C data." Bioinformatics 35.21 (2019): 4222-4228.

[5] Hong, Hao, et al. "DeepHiC: A generative adversarial network for enhancing Hi-C data resolution." PLoS computational biology 16.2 (2020): e1007287.

**There are not too many deep learning models for enhancing Hi-C resolution.**
 
### Non-deep-learning models
[6] Zhang, Shilu, et al. "In silico prediction of high-resolution Hi-C interaction matrices." Nature communications 10.1 (2019): 1-18.

**This method solves the problem from another perspective using traditional machine learning models such as a random forest. There may be other non-deep-learning models. You could search for it. I think matrix completion (imputation) can be a good start.**

***

### Downstream analysis tasks over Hi-C data
[7] Zhang, Yan. "Investigate Genomic 3D Structure Using Deep Neural Network." (2017).

**This is the dissertation of the author of HiCPlus. This dissertation introduce some downstream anlysis in details.**

### Other resources that might be related
[8] Li, Wenyuan, et al. "Hi-Corrector: a fast, scalable and memory-efficient package for normalizing large-scale Hi-C data." Bioinformatics 31.6 (2015): 960-962.

[9] [Data Production and Processing Standard of the Hi-C Mapping Center] (https://www.encodeproject.org/documents/75926e4b-77aa-4959-8ca7-87efcba39d79/@@download/attachment/comp_doc_7july2018_final.pdf)

[10] Djekidel, Mohamed Nadhir, Yang Chen, and Michael Q. Zhang. "FIND: difFerential chromatin INteractions Detection using a spatial Poisson process." Genome research 28.3 (2018): 412-422.



## TODO:
[1] Literatue Review (due at 11.59 pm Friday)

