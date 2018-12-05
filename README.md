## ml-in-cosmology
This github attempts to maintain a comprehensive list of published machine learning applications to cosmology, organized by subject matter and arxiv posting date. Each entry contains the paper title, a simple summary of the machine learning methods used in the work, and the arxiv link. If I have missed any cosmology papers that you believe should be included please email me at gstein@cita.utoronto.ca or issue a pull request. 

I am currently a PhD Candidate at the Canadian Institute for Theoretical astrophysics, broadly working on problems in computational cosmology, but with a great interest in machine learning methods, and just made this for fun and to help anyone with similar interests. More info about me can be found at https://www.cita.utoronto.ca/~gstein/, and cheers to whoever can find which one of the papers below is mine :beers: 

## Table of Contents
---
#### Section List
- [Dictionary](#dictionary)
- [Large-Scale Structure](#structure)
  - [Formation](#formation)
  - [Identification](#identification)
- [Reionization and 21cm](#reionization21cm)
- [Gravitational Lensing](#lensing)
  - [Weak Lensing](#weak)
  - [Strong Lensing](#strong)
- [Cosmic Microwave Background](#cmb)
- [Observational](#observational)
- [Tools](#tools)
- [Reviews](#reviews)



<a name='dictionary'></a>
#### Dictionary

A dictionary of all abbreviations for machine learning methods used in this compilation. In general I adopted those used by the authors, except in a few cases. The links are to explanatory articles that I personally like. 
 
* ADA: [AdaBoosted](https://en.wikipedia.org/wiki/AdaBoost) [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree_learning)
* AdR: [AdaBoostRegressor](https://en.wikipedia.org/wiki/AdaBoost)
* BRR: [Bayesian](https://en.wikipedia.org/wiki/Bayesian_linear_regression) [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) [Regression](https://onlinecourses.science.psu.edu/stat857/node/155/)
* CNN: [Convolutional Neural Network](https://cs231n.github.io/convolutional-networks/)
* DCMDN: [Deep Convolutional Mixture Density Network](https://www.h-its.org/ain-software-en/deep-convolutional-mixture-density-networks/)
* DT: [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree_learning)
* ET: [Extremely Randomized Trees](https://link.springer.com/article/10.1007/s10994-006-6226-1)
* GAN: [Generative](https://towardsdatascience.com/generative-adversarial-networks-gans-a-beginners-guide-5b38eceece24) [Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets)
* GBDT: [Gradient Boosted Regressor Trees](https://en.wikipedia.org/wiki/Gradient_boosting)
* GPR: [Gaussian Process Regression](https://www.mathworks.com/help/stats/gaussian-process-regression-models.html)
* kNN: [k-Nearest Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* KRR: [Kernel](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf) [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) [Regression](https://onlinecourses.science.psu.edu/stat857/node/155/)
* MDN:  [Mixture](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf) [Density Network](https://cbonnett.github.io/MDN.html)
* MLP: [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
* NN: [Neural](https://en.wikipedia.org/wiki/Artificial_neural_network) [Network](https://cs231n.github.io/neural-networks-1/)
* OLR: [Ordinary Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
* RF: [Random Forests](https://en.wikipedia.org/wiki/Random_forest)
* RNN: [Recurrent](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) [Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
* RR: [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) [Regression](https://onlinecourses.science.psu.edu/stat857/node/155/)
* SVR: [Support Vector Regression](https://en.wikipedia.org/wiki/Support_vector_machine)
* V-Net: [three dimensional](https://arxiv.org/abs/1606.04797) [U-Net](http://deeplearning.net/tutorial/unet.html)


&nbsp;

<a name='structure'></a>
## Large-Scale Structure
---

The [Large-Scale structure](https://ned.ipac.caltech.edu/level5/March12/Coil/Coil1.html) of the universe is a field that relies on state-of-the art cosmological simulations to address a number of questions. Due to the computational complexity of these simulations, some investigations will remain computationally-infeasible for the forseeable future, and machine learning techniques can have a number of important uses.
<a name='formation'></a>
#### Structure Formation

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *PkANN I&2. Non-linear matter power spectrum interpolation through artificial neural networks* | NN | https://arxiv.org/abs/1203.1695, https://arxiv.org/abs/1312.2101 |
| *Machine learning and cosmological simulations I.&II.* | kNN, DT, RF, ET | https://arxiv.org/abs/1510.06402 https://arxiv.org/abs/1510.07659 |
| *Painting galaxies into dark matter haloes using machine learning* | SVR, kNN, MLP, DT, RF, ET, AdR | https://arxiv.org/abs/1712.03255 |
| *Modeling the Impact of Baryons on Subhalo Populations with Machine Learning* | RF | https://arxiv.org/abs/1712.04467 |
| *Fast cosmic web simulations with generative adversarial networks* | GAN | https://arxiv.org/abs/1801.09070 |
| *Machine learning cosmological structure formation* | RF | https://arxiv.org/abs/1802.04271 | 
| *Classifying the Large Scale Structure of the Universe with Deep Neural Networks* | V-Net | https://arxiv.org/abs/1804.00816 |
| *A volumetric deep Convolutional Neural Network for simulation of mock dark matter halo catalogues* | V-Net | https://arxiv.org/abs/1805.04537 |
| *Learning to Predict the Cosmological Structure Formation* | V-Net | https://arxiv.org/abs/1811.06533 |

<a name='identification'></a>
#### Structure Identification
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *An application of machine learning techniques to galaxy cluster mass estimation using the MACSIS simulations* | OLR, RR, BRR, KRR, SVR, DT, GBDT, ADA, kNN | https://arxiv.org/abs/1810.08430 |


&nbsp;

<a name='reionization21cm'></a>
## Reionization and 21cm
---
In cosmology, the process of [Reionization](https://en.wikipedia.org/wiki/Reionization) refers to the period when our universe went from the "Dark Ages" before major star and galaxy formation, to the ionized (neutral Hydrogen) state we see today. 

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A machine-learning approach to measuring the escape of ionizing radiation from galaxies in the reionization epoch* | LR | https://arxiv.org/abs/1603.09610 | 
| *Analysing the 21 cm signal from the epoch of reionization with artificial neural networks* | NN | https://arxiv.org/abs/1701.07026 |
| *Emulation of reionization simulations for Bayesian inference of astrophysics parameters using neural networks* | NN | https://arxiv.org/abs/1708.00011 | 
| *Reionization Models Classifier using 21cm Map Deep Learning* | CNN | https://arxiv.org/abs/1801.06381 |
| *Deep learning from 21-cm images of the Cosmic Dawn* | CNN | https://arxiv.org/abs/1805.02699 | 
| *Evaluating machine learning techniques for predicting power spectra from reionization simulations* | SVM, MLP, GPR | https://arxiv.org/abs/1811.09141 |




&nbsp;

<a name='lensing'></a>
## Gravitational Lensing
---
[Gravitational lensing](http://www.cfhtlens.org/public/what-gravitational-lensing) in cosmology refers by the bending of light due to mass between the source and Earth. This effect is very useful for inferring properties of the total mass distribution in our Universe, which is dominated by dark matter. Gravitational Lensing comes in two types: weak and strong.

Strong gravitational lensing refers to the cases where the lensing effect ( e.g. multiple images, clear shape distortions) is strong enough to be seen by the human eye, or equivalent, on an astronomical image. This only happens when a massive galaxy cluster lies between us and some background galaxies

Weak gravitational lensing refers to the global effect that almost all far away galaxies are gravitationally lensed by a small amount, which changes their observed shape by roughly 1%. This can only be measured statistically when given a large number of observations, and not on an object-to-object basis.

<a name='weak'></a>
#### Weak Lensing
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *Creating Virtual Universes Using Generative Adversarial Networks* | GAN | https://arxiv.org/abs/1706.02390 |
| *Cosmological model discrimination with Deep Learning* | CNN | https://arxiv.org/abs/1707.05167 |
| *Non-Gaussian information from weak lensing data via deep learning* | CNN | https://arxiv.org/abs/1802.01212 |
| *Learning from deep learning: better cosmological parameter inference from weak lensing maps* | CNN | https://arxiv.org/abs/1806.05995 |
| *Cosmological constraints from noisy convergence maps through deep learning* | CNN | https://arxiv.org/abs/1807.08732 |
| *On the dissection of degenerate cosmologies with machine learning* | CNN | https://arxiv.org/abs/1810.11027 |
| *Distinguishing standard and modified gravity cosmologies with machine learning | CNN | https://arxiv.org/abs/1810.11030 | 

<a name='strong'></a>
#### Strong Lensing
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A neural network gravitational arc finder based on the Mediatrix filamentation method* | NN | https://arxiv.org/abs/1607.04644 |
| *CMU DeepLens: deep learning for automatic image-based galaxy-galaxy strong lens finding* | CNN | https://arxiv.org/abs/1703.02642 |
| *Finding strong lenses in CFHTLS using convolutional neural networks* | CNN | https://arxiv.org/abs/1704.02744 |
| *Fast automated analysis of strong gravitational lenses with convolutional neural networks* | CNN | https://arxiv.org/abs/1708.08842 |
| *Uncertainties in Parameters Estimated with Neural Networks: Application to Strong Gravitational Lensing* | NN | https://arxiv.org/abs/1708.08843 |
| *Testing convolutional neural networks for finding strong gravitational lenses in KiDS* | CNN | https://arxiv.org/abs/1807.04764 |
| *Analyzing interferometric observations of strong gravitational lenses with recurrent and convolutional neural networks* | RNN, CNN | https://arxiv.org/abs/1808.00011 |


&nbsp;

<a name='cmb'></a>
## Cosmic Microwave Background
---
The [Cosmic Microwave Background (CMB)](https://en.wikipedia.org/wiki/Cosmic_microwave_background) is the light left over from the period of recombination in the very early Universe, 380,000 years after the beginning. CMB observations are sometimes referred to as "baby pictures of our Universe", as this light has been travelling for 13.5 billion years just to reach us.

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *DeepCMB: Lensing Reconstruction of the Cosmic Microwave Background with Deep Neural Networks* | CNN | https://arxiv.org/abs/1810.01483 |




&nbsp;

<a name='observational'></a>
## Observational
---

This section has a variety of machine learning papers used for various other observational applications.

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *Robust Machine Learning Applied to Astronomical Data Sets. II. Quantifying Photometric Redshifts for Quasars Using Instance-based Learning* | kNN | https://arxiv.org/abs/astro-ph/0612471 |
| *Robust Machine Learning Applied to Astronomical Data Sets. III. Probabilistic Photometric Redshifts for Galaxies and Quasars in the SDSS and GALEX* | kNN | https://arxiv.org/abs/0804.3413 |
| *Photometric redshift estimation via deep learning. Generalized and pre-classification-less, image based, fully probabilistic redshifts* | RF, MDN, DCMDN | https://arxiv.org/abs/1706.02467 |
| *Deep Learning of Quasar Spectra to Discover and Characterize Damped Lya Systems* | CNN | https://arxiv.org/abs/1709.04962 | 
| *Predicting the Neutral Hydrogen Content of Galaxies From Optical Data Using Machine Learning* | OLR, RF, GBDT, kNN, SVM, NN | https://arxiv.org/abs/1803.08334 |
| *Knowledge transfer of Deep Learning for galaxy morphology from one survey to another* | CNN | https://arxiv.org/abs/1807.00807 |



&nbsp;

<a name='tools'></a>
## Tools
---
Contained here are some machine learning tools that are specifically designed for the computational challenges of cosmology. 

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *CosmoFlow: Using Deep Learning to Learn the Universe at Scale* | CNN | https://arxiv.org/abs/1808.04728 |


&nbsp;

<a name='reviews'></a>
## Reviews
---
Reviews of machine learning in cosmology, and, more broadly, astronomy.
| Title | arxiv link |
| :---  | :---: |
| *Data Mining and Machine Learning in Astronomy* | https://arxiv.org/abs/0906.2173 | 