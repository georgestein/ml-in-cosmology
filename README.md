## ml-in-cosmology
This github attempts to maintain a comprehensive list of published machine learning applications to cosmology, organized by subject matter and arxiv posting date. Each entry contains the paper title, a simple summary of the machine learning methods used in the work, and the arxiv link. If I have missed any cosmology papers that you believe should be included please email me at gstein@berkeley.edu or issue a pull request. 

Feel free to cite in any works
[![DOI](https://zenodo.org/badge/160429652.svg)](https://zenodo.org/badge/latestdoi/160429652)

I am currently a postdoctoral researcher at the Berkeley Center for Cosmological Physics, broadly working on problems in computational cosmology, but with a great interest in machine learning methods, and just made this for fun and to help anyone with similar interests. More info about me can be found at https://www.cita.utoronto.ca/~gstein/, and cheers to whoever can find which of the papers below have me as an author :beers: 

---
## Table of Contents
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
  - [Redshift Prediction](#redshifts)
  - [Other](#otherobservational)
- [Parameter Estimation](#parameters)
- [Tools](#tools)
- [Public Datasets](#datasets)
- [Reviews](#reviews)
- [Acknowledgments](#acknowledgments)



<a name='dictionary'></a>
#### Dictionary

A dictionary of all abbreviations for machine learning methods used in this compilation. In general I adopted those used by the authors, except in a few cases. The links are to explanatory articles that I personally like. 
 
* ADA: [AdaBoosted](https://en.wikipedia.org/wiki/AdaBoost) [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree_learning)
* AdR: [AdaBoostRegressor](https://en.wikipedia.org/wiki/AdaBoost)
* BDT: [Boosted Decision Tree](https://en.wikipedia.org/wiki/Gradient_boosting)
* BNN: [Bayesian Neural Network](https://krasserm.github.io/2019/03/14/bayesian-neural-networks/)
* BRR: [Bayesian](https://en.wikipedia.org/wiki/Bayesian_linear_regression) [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) [Regression](https://onlinecourses.science.psu.edu/stat857/node/155/)
* CNN: [Convolutional Neural Network](https://cs231n.github.io/convolutional-networks/)
* DCMDN: [Deep Convolutional Mixture Density Network](https://www.h-its.org/ain-software-en/deep-convolutional-mixture-density-networks/)
* DT: [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree_learning)
* EXT: [Extremely Randomized Trees](https://link.springer.com/article/10.1007/s10994-006-6226-1)
* GA: [Genetic Algorithm](https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/)
* GAN: [Generative](https://towardsdatascience.com/generative-adversarial-networks-gans-a-beginners-guide-5b38eceece24) [Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets)
* GMM: [Gaussian Mixture Model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
* GPR: [Gaussian Process Regression](https://www.mathworks.com/help/stats/gaussian-process-regression-models.html)
* HNN: [Hopfield Neural Network](https://www.doc.ic.ac.uk/~sd4215/hopfield.html)
* kNN: [k-Nearest Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* KRR: [Kernel](https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf) [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) [Regression](https://onlinecourses.science.psu.edu/stat857/node/155/)
* NPE: Neural Physical Engine
* MAF: [Masked Autoregressive Flows](https://arxiv.org/pdf/1705.07057.pdf)
* MADE: [Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509)
* MINT:  [Mutual Information based Transductive Feature Selection](https://arxiv.org/abs/1310.1659)
* MDN:  [Mixture](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf) [Density Network](https://cbonnett.github.io/MDN.html)
* MLP: [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
* NN: [Neural](https://en.wikipedia.org/wiki/Artificial_neural_network) [Network](https://cs231n.github.io/neural-networks-1/)
* NF: [Normalizing Flow](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
* OLR: [Ordinary Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
* RF: [Random Forests](https://en.wikipedia.org/wiki/Random_forest)
* RIM: [Recurrent Interence Machines](https://arxiv.org/abs/1706.04008)
* RNN: [Recurrent](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) [Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
* RR: [Ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) [Regression](https://onlinecourses.science.psu.edu/stat857/node/155/)
* SOM: [Self Organising](https://en.wikipedia.org/wiki/Self-organizing_map) [Map](https://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html)
* SVM: [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
* SVR: [Support Vector Regression](https://en.wikipedia.org/wiki/Support_vector_machine)
* TINT: [Tree-Interpreter](https://github.com/andosa/treeinterpreter)
* V-Net: [three dimensional](https://arxiv.org/abs/1606.04797) [U-Net](http://deeplearning.net/tutorial/unet.html)
* XGBoost: [eXtreme Gradient Boosting](https://xgboost.readthedocs.io/en/latest/)

&nbsp;

---
<a name='structure'></a>
## Large-Scale Structure


The [Large-Scale Structure](https://ned.ipac.caltech.edu/level5/March12/Coil/Coil1.html) of the universe is a field that relies on state-of-the art cosmological simulations to address a number of questions. Due to the computational complexity of these simulations, some investigations will remain computationally-infeasible for the forseeable future, and machine learning techniques can have a number of important uses.
<a name='formation'></a>
#### Structure Formation

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A First Look at creating mock catalogs with machine learning techniques* | SVM, kNN | https://arxiv.org/abs/1303.1055 |
| *Machine Learning Etudes in Astrophysics: Selection Functions for Mock Cluster Catalogs* | SVM, GMM | https://arxiv.org/abs/1409.1576 |
| *PkANN I&2. Non-linear matter power spectrum interpolation through artificial neural networks* | NN | https://arxiv.org/abs/1203.1695, https://arxiv.org/abs/1312.2101 |
| *Machine learning and cosmological simulations I.&II.* | kNN, DT, RF, EXT | https://arxiv.org/abs/1510.06402 https://arxiv.org/abs/1510.07659 |
| *Estimating Cosmological Parameters from the Dark Matter Distribution* | CNN | https://arxiv.org/abs/1711.02033 |
| *Painting galaxies into dark matter haloes using machine learning* | SVR, kNN, MLP, DT, RF, EXT, AdR | https://arxiv.org/abs/1712.03255 |
| *Modeling the Impact of Baryons on Subhalo Populations with Machine Learning* | RF | https://arxiv.org/abs/1712.04467 |
| *Fast cosmic web simulations with generative adversarial networks* | GAN | https://arxiv.org/abs/1801.09070 |
| *Machine learning cosmological structure formation* | RF | https://arxiv.org/abs/1802.04271 | 
| *A Machine Learning Approach to Galaxy-LSS Classification I: Imprints on Halo Merger Trees* | SVM | https://arxiv.org/abs/1803.11156 |
| *Classifying the Large Scale Structure of the Universe with Deep Neural Networks* | V-Net | https://arxiv.org/abs/1804.00816 |
| *Cosmological Reconstruction From Galaxy Light: Neural Network Based Light-Matter Connection*| NN | https://arxiv.org/abs/1805.02247 |
| *A volumetric deep Convolutional Neural Network for simulation of mock dark matter halo catalogues* | V-Net | https://arxiv.org/abs/1805.04537 |
| *Learning to Predict the Cosmological Structure Formation* | V-Net | https://arxiv.org/abs/1811.06533 |
| *deepCool: Fast and Accurate Estimation of Cooling Rates in Irradiated Gas with Artificial Neural Networks* | NN, RF, kNN | https://arxiv.org/abs/1901.01264 |
| *From Dark Matter to Galaxies with Convolutional Networks* | V-Net | https://arxiv.org/abs/1902.05965 | 
| *Painting halos from 3D dark matter fields using Wasserstein mapping networks* | GAN | https://arxiv.org/abs/1903.10524 |
| *Painting with baryons: augmenting N-body simulations with gas using deep generative models* | GAN, VAE | https://arxiv.org/abs/1903.12173 |
| *HIGAN: Cosmic Neutral Hydrogen with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/1904.12846 |
| *A deep learning model to emulate simulations of cosmic reionization* | CNN | https://arxiv.org/abs/1905.06958 |
| *An interpretable machine learning framework for dark matter halo formation* | BDT | https://arxiv.org/abs/1906.06339 | 
| *Cosmological N-body simulations: a challenge for scalable generative models* | GAN | https://arxiv.org/abs/1908.05519 |
| *Cosmological parameter estimation from large-scale structure deep learning* | CNN | https://arxiv.org/abs/1908.10590 |
| *Neural physical engines for inferring the halo mass distribution function* | NPE | https://arxiv.org/abs/1909.06379 |
| *A Hybrid Deep Learning Approach to Cosmological Constraints From Galaxy Redshift Surveys* | CNN | https://arxiv.org/abs/1909.10527 |
| *A black box for dark sector physics: Predicting dark matter annihilation feedback with conditional GANs* | cGAN | https://arxiv.org/abs/1910.00291 |
| *Learning neutrino effects in Cosmology with Convolutional Neural Networks* | V-Net |https://arxiv.org/abs/1910.04255 |
| *Predicting dark matter halo formation in N-body simulations with deep regression networks* | V-Net | https://arxiv.org/abs/1912.04299 |
| *Probabilistic cosmic web classification using fast-generated training data* | RF | https://arxiv.org/abs/1912.04412 |
| *Super-resolution emulator of cosmological simulations using deep physical models* | WGAN | https://arxiv.org/abs/2001.05519 |
| *Baryon acoustic oscillations reconstruction using convolutional neural networks* | CNN | https://arxiv.org/abs/2002.10218 |
| *Emulation of cosmological mass maps with conditional generative adversarial networks* | GAN | https://arxiv.org/abs/2004.08139 |
| *Towards Universal Cosmological Emulators with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/2004.10223 |
| *Nonlinear 3D Cosmic Web Simulation with Heavy-Tailed Generative Adversarial Networks* | GAN | https://arxiv.org/abs/2005.03050 |
| *GalaxyNet: Connecting galaxies and dark matter haloes with deep neural networks and reinforcement learning in large volumes* | RF, NN | https://arxiv.org/abs/2005.12276 |
| *Discovering Symbolic Models from Deep Learning with Inductive Biases* | GNN | https://arxiv.org/abs/2006.11287 |
| *Teaching neural networks to generate Fast Sunyaev Zel'dovich Maps* | V-Net | https://arxiv.org/abs/2007.07267 |
| *HInet: Generating neutral hydrogen from dark matter with neural networks* | CNN | https://arxiv.org/abs/2007.10340 |
| *Machine Learning the Fates of Dark Matter Subhalos: A Fuzzy Crystal Ball* | RF, BDT | https://arxiv.org/abs/2008.05001 |  
| *Learning effective physical laws for generating cosmological hydrodynamics with Lagrangian Deep Learning* | LDL | https://arxiv.org/abs/2010.02926 |

<a name='identification'></a>
#### Structure Identification
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A Machine Learning Approach for Dynamical Mass Measurements of Galaxy Clusters* | SDM | https://arxiv.org/abs/1410.0686, https://arxiv.org/abs/1509.05409 |
| *A Deep Learning Approach to Galaxy Cluster X-ray Masses* | CNN | https://arxiv.org/abs/1810.07703 |
| *An application of machine learning techniques to galaxy cluster mass estimation using the MACSIS simulations* | OLR, RR, BRR, KRR, SVR, DT, BDT, ADA, kNN | https://arxiv.org/abs/1810.08430 |
| *Prediction of galaxy halo masses in SDSS DR7 via a machine learning approach* | XGBoost, RF, NN | https://arxiv.org/abs/1902.02680 | 
| *A Robust and Efficient Deep Learning Method for Dynamical Mass Measurements of Galaxy Clusters* | CNN | https://arxiv.org/abs/1902.05950 |
| *Multiwavelength cluster mass estimates and machine learning* | GB, RF | https://arxiv.org/abs/1905.09920 |
| *Using X-Ray Morphological Parameters to Strengthen Galaxy Cluster Mass Estimates via Machine Learning* | RF | https://arxiv.org/abs/1908.02765 |
| *Large-scale structures in the LCDM Universe: network analysis and machine learning* | XGBoost | https://arxiv.org/abs/1910.07868 |
| *Dynamical mass inference of galaxy clusters with neural flows* | NF (MADE) | https://arxiv.org/abs/2003.05951 |
| *Mass Estimation of Galaxy Clusters with Deep Learning I: Sunyaev-Zel'dovich Effect* | U-Net | https://arxiv.org/abs/2003.06135 |
| *Galaxy cluster mass estimation with deep learning and hydrodynamical simulations* | CNN | https://arxiv.org/abs/2005.11819 |
| *Mass Estimation of Galaxy Clusters with Deep Learning II: CMB Cluster Lensing* | U-NET | https://arxiv.org/abs/2005.13985 |
| *Multiwavelength classification of X-ray selected galaxy cluster candidates using convolutional neural networks* | CNN | https://arxiv.org/abs/2006.05998 |
| *Approximate Bayesian Uncertainties on Deep Learning Dynamical Mass Estimates of Galaxy Clusters* | BNN | https://arxiv.org/abs/2006.13231 |
| *A deep learning view of the census of galaxy clusters in IllustrisTNG* | CNN | https://arxiv.org/abs/2007.05144 |
| *Revealing the Local Cosmic Web by Deep Learning* | V-Net | https://arxiv.org/abs/2008.01738 |
| *Simulation-based inference of dynamical galaxy cluster masses with 3D convolutional neural networks* | CNN | https://arxiv.org/abs/2009.03340 |
&nbsp;

---
<a name='reionization21cm'></a>
## Reionization and 21cm
In cosmology, the process of [Reionization](https://en.wikipedia.org/wiki/Reionization) refers to the period when our universe went from the "Dark Ages" before major star and galaxy formation, to the ionized state we see today. 

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A machine-learning approach to measuring the escape of ionizing radiation from galaxies in the reionization epoch* | LR | https://arxiv.org/abs/1603.09610 | 
| *Analysing the 21 cm signal from the epoch of reionization with artificial neural networks* | NN | https://arxiv.org/abs/1701.07026 |
| *Emulation of reionization simulations for Bayesian inference of astrophysics parameters using neural networks* | NN | https://arxiv.org/abs/1708.00011 | 
| *Reionization Models Classifier using 21cm Map Deep Learning* | CNN | https://arxiv.org/abs/1801.06381 |
| *Deep learning from 21-cm images of the Cosmic Dawn* | CNN | https://arxiv.org/abs/1805.02699 | 
| *Identifying Reionization Sources from 21cm Maps using Convolutional Neural Networks* | CNN | https://arxiv.org/abs/1807.03317 |
| *Evaluating machine learning techniques for predicting power spectra from reionization simulations* | SVM, MLP, GPR | https://arxiv.org/abs/1811.09141 |
| *Improved supervised learning methods for EoR parameters reconstruction* | CNN | https://arxiv.org/abs/1904.04106 |
| *Constraining the astrophysics and cosmology from 21cm tomography using deep learning with the SKA*| CNN | https://arxiv.org/abs/1907.07787 | 
| *21cm Global Signal Extraction: Extracting the 21cm Global Signal using Artificial Neural Networks* | NN | https://arxiv.org/abs/1911.02580 |
| *A unified framework for 21cm tomography sample generation and parameter inference with Progressively Growing GANs* | GAN | https://arxiv.org/abs/2002.07940 |
| *Beyond the power spectrum - I: recovering H II bubble size distribution from 21 cm power spectrum with artificial neural networks* | NN | https://arxiv.org/abs/2002.08238 |
| *Foreground modelling via Gaussian process regression: an application to HERA data* | GP | https://arxiv.org/abs/2004.06041 |
| *Predicting 21cm-line map from Lyman α emitter distribution with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/2004.09206 |
| *Constraining the Reionization History using Bayesian Normalizing Flows* | NF | https://arxiv.org/abs/2005.07694 | 
| *Deep-Learning Study of the 21cm Differential Brightness Temperature During the Epoch of Reionization* | CNN | https://arxiv.org/abs/2006.06236 |
| *Removing Astrophysics in 21 cm maps with Neural Networks* | CNN | https://arxiv.org/abs/2006.14305 |

&nbsp;
---
<a name='lensing'></a>
## Gravitational Lensing
[Gravitational lensing](http://www.cfhtlens.org/public/what-gravitational-lensing) in cosmology refers to the bending of light due to mass between the source and Earth. This effect is very useful for inferring properties of the total mass distribution in our Universe, which is dominated by dark matter that we cannot see electromagnetically. Gravitational lensing comes in two types: weak and strong.

Strong gravitational lensing refers to the cases where the lensing effect (e.g. multiple images, clear shape distortions) is strong enough to be seen by the human eye, or equivalent, on an astronomical image. This only happens when a massive galaxy cluster lies between us and some background galaxies

Weak gravitational lensing refers to the global effect that almost all far away galaxies are gravitationally lensed by a small amount, which changes their observed shape by roughly 1%. This can only be measured statistically when given a large number of samples, and not on an object-to-object basis.

<a name='weak'></a>
#### Weak Lensing
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *Bias-Free Shear Estimation using Artificial Neural Networks* | NN | https://arxiv.org/abs/1002.0838 |
| *Hopfield Neural Network deconvolution for weak lensing measurement* | HNN | https://arxiv.org/abs/1411.3193 |
| *CosmoGAN: creating high-fidelity weak lensing convergence maps using Generative Adversarial Networks* | GAN | https://arxiv.org/abs/1706.02390 |
| *Cosmological model discrimination with Deep Learning* | CNN | https://arxiv.org/abs/1707.05167 |
| *Non-Gaussian information from weak lensing data via deep learning* | CNN | https://arxiv.org/abs/1802.01212 |
| *Learning from deep learning: better cosmological parameter inference from weak lensing maps* | CNN | https://arxiv.org/abs/1806.05995 |
| *Weak-lensing shear measurement with machine learning: teaching artificial neural networks about feature noise* | NN | https://arxiv.org/abs/1807.02120 |
| *Cosmological constraints from noisy convergence maps through deep learning* | CNN | https://arxiv.org/abs/1807.08732 |
| *Weak lensing shear estimation beyond the shape-noise limit: a machine learning approach* | CNN | https://arxiv.org/abs/1808.07491 |
| *On the dissection of degenerate cosmologies with machine learning* | CNN | https://arxiv.org/abs/1810.11027 |
| *Distinguishing standard and modified gravity cosmologies with machine learning* | CNN | https://arxiv.org/abs/1810.11030 |
| *Denoising Weak Lensing Mass Maps with Deep Learning* | GAN | https://arxiv.org/abs/1812.05781 |
| *Weak lensing cosmology with convolutional neural networks on noisy data* | CNN | https://arxiv.org/abs/1902.03663 |
| *Galaxy shape measurement with convolutional neural networks* | CNN | https://arxiv.org/abs/1902.08161 |
| *Cosmological constraints with deep learning from KiDS-450 weak lensing maps* | CNN | https://arxiv.org/abs/1906.03156 |
| *Deep learning dark matter map reconstructions from DES SV weak lensing data* | U-Net | https://arxiv.org/abs/1908.00543 |
| *Decoding Cosmological Information in Weak-Lensing Mass Maps with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/1911.12890 |
| *Parameter Inference for Weak Lensing using Gaussian Processes and MOPED* | GP | https://arxiv.org/abs/2005.06551 |
| *Shear measurement bias II: a fast machine learning calibration method* | NN | https://arxiv.org/abs/2006.07011 |
| *Interpreting deep learning models for weak lensing* | CNN | https://arxiv.org/abs/2007.06529 |

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
| *Data-Driven Reconstruction of Gravitationally Lensed Galaxies using Recurrent Inference Machines* | RIM, CNN | https://arxiv.org/abs/1901.01359 |
| *Mining for Dark Matter Substructure: Inferring subhalo population properties from strong lenses with machine learning* | NN | https://arxiv.org/abs/1909.02005 |
| *Deep Learning the Morphology of Dark Matter Substructure* | CNN | https://arxiv.org/abs/1909.07346 |
| *Circumventing Lens Modeling to Detect Dark Matter Substructure in Strong Lens Images with Convolutional Neural Networks* | CNN | https://arxiv.org/abs/1910.00015 |
| *Differentiable Strong Lensing: Uniting Gravity and Neural Nets through Differentiable Probabilistic Programming* | VAE | https://arxiv.org/abs/1910.06157 |
| *Identifying Strong Lenses with Unsupervised Machine Learning using Convolutional Autoencoder* | VAE | https://arxiv.org/abs/1911.04320 || *Modular Deep Learning Analysis of Galaxy-Scale Strong Lensing Images* | CNN | https://arxiv.org/abs/1911.03867 | 
| *HOLISMOKES II. Identifying galaxy-scale strong gravitational lenses in Pan-STARRS using convolutional neural networks* | CNN | https://arxiv.org/abs/2004.13048 | 
| *Dark Matter Subhalos, Strong Lensing and Machine Learning* | CNN | https://arxiv.org/abs/2005.05353 |
| *Deep Learning for Strong Lensing Search: Tests of the Convolutional Neural Networks and New Candidates from KiDS DR3* | CNN | https://arxiv.org/abs/2007.00188 |
| *Extracting the Subhalo Mass Function from Strong Lens Images with Image Segmentation* | U-Net | https://arxiv.org/abs/2009.06639 | 
| *Detecting Subhalos in Strong Gravitational Lens Images with Image Segmentation* | U-Net | https://arxiv.org/abs/2009.06663 |

&nbsp;

---
<a name='cmb'></a>
## Cosmic Microwave Background
The [Cosmic Microwave Background (CMB)](https://en.wikipedia.org/wiki/Cosmic_microwave_background) is the light left over from the period of recombination in the very early Universe, 380,000 years after the beginning. CMB observations are sometimes referred to as "baby pictures of our Universe", as this light has been travelling for 13.5 billion years just to reach us.

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *DeepCMB: Lensing Reconstruction of the Cosmic Microwave Background with Deep Neural Networks* | CNN | https://arxiv.org/abs/1810.01483 |
| *Fast Wiener filtering of CMB maps with Neural Networks* | U-Net | https://arxiv.org/abs/1905.05846 |
| *CMB-GAN: Fast Simulations of Cosmic Microwave background anisotropy maps using Deep Learning* | GAN | https://arxiv.org/abs/1908.04682 |
| *CosmoVAE: Variational Autoencoder for CMB Image Inpainting* | VAE | https://arxiv.org/abs/2001.11651 |
| *Inpainting Galactic Foreground Intensity and Polarization maps using Convolutional Neural Network* | GAN | https://arxiv.org/abs/2003.13691 |
| *Inpainting via Generative Adversarial Networks for CMB data analysis* | GAN | https://arxiv.org/abs/2004.04177 |
| *Full-sky Cosmic Microwave Background Foreground Cleaning Using Machine Learning* | BNN | https://arxiv.org/abs/2004.11507 |
| *Foreground model recognition through Neural Networks for CMB B-mode observations* | NN | https://arxiv.org/abs/2003.02278 |

&nbsp;

---
<a name='observational'></a>
## Observational

This section has a variety of machine learning papers used for various observational applications.

<a name='redshifts'></a>
#### Redshifts
This section is definitely not exhaustive - there is a massive amount of work in this subject area.  

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *ANNz: estimating photometric redshifts using artificial neural networks* | NN | https://arxiv.org/abs/astro-ph/0311058 | 
| *Estimating Photometric Redshifts Using Support Vector Machines* | SVM | https://arxiv.org/abs/astro-ph/0412005 |
| *Robust Machine Learning Applied to Astronomical Data Sets. II. Quantifying Photometric Redshifts for Quasars Using Instance-based Learning* | kNN | https://arxiv.org/abs/astro-ph/0612471 |
| *Robust Machine Learning Applied to Astronomical Data Sets. III. Probabilistic Photometric Redshifts for Galaxies and Quasars in the SDSS and GALEX* | kNN | https://arxiv.org/abs/0804.3413 |
| *ArborZ: Photometric Redshifts Using Boosted Decision Trees* | BDT | https://arxiv.org/abs/0908.4085 |
| *Unsupervised self-organised mapping: a versatile empirical tool for object selection, classification and redshift estimation in large surveys* | SOM | https://arxiv.org/abs/1110.0005 | 
| *Can Self-Organizing Maps accurately predict photometric redshifts?* | SOM | https://arxiv.org/abs/1201.1098 |
| *TPZ : Photometric redshift PDFs and ancillary information by using prediction trees and random forests* | RF | https://arxiv.org/abs/1303.7269 | 
| *Estimating Photometric Redshifts of Quasars via K-nearest Neighbor Approach Based on Large Survey Databases* | kNN | https://arxiv.org/abs/1305.5023 |
| *An approach to the analysis of SDSS spectroscopic outliers based on Self-Organizing Maps* | SOM | https://arxiv.org/abs/1309.2418 | 
| *Using neural networks to estimate redshift distributions. An application to CFHTLenS* | NN | https://arxiv.org/abs/1312.1287 |
| *SOMz: photometric redshift PDFs with self organizing maps and random atlas* | SOM | https://arxiv.org/abs/1312.5753 |
| *Feature importance for machine learning redshifts applied to SDSS galaxies* | NN, ADA | https://arxiv.org/abs/1410.4696 |
| *GAz: A Genetic Algorithm for Photometric Redshift Estimation* | GA | https://arxiv.org/abs/1412.5997 |
| *Anomaly detection for machine learning redshifts applied to SDSS galaxies* | ADA, SOM, BDT | https://arxiv.org/abs/1503.08214 |
| *Measuring photometric redshifts using galaxy images and Deep Neural Networks* | CNN, ADA | https://arxiv.org/abs/1504.07255 |
| *A Sparse Gaussian Process Framework for Photometric Redshift Estimation* | NN, GPR | https://arxiv.org/abs/1505.05489 | 
| *ANNz2 - photometric redshift and probability distribution function estimation using machine learning* | NN, BDT | https://arxiv.org/abs/1507.00490 | 
| *DNF - Galaxy photometric redshift by Directional Neighbourhood Fitting* | kNN | https://arxiv.org/abs/1511.07623 | 
| *Photometric Redshift Estimation for Quasars by Integration of KNN and SVM* | kNN, SVM | https://arxiv.org/abs/1601.01739 |
| *Stacking for machine learning redshifts applied to SDSS galaxies* | SOM, DT | https://arxiv.org/abs/1602.06294 |
| *GPz: Non-stationary sparse Gaussian processes for heteroscedastic uncertainty estimation in photometric redshifts* | GPR | https://arxiv.org/abs/1604.03593 |
| *Photo-z with CuBANz: An improved photometric redshift estimator using Clustering aided Back Propagation Neural network* | NN | https://arxiv.org/abs/1609.03568 |
| *Photometric redshift estimation via deep learning. Generalized and pre-classification-less, image based, fully probabilistic redshifts* | RF, MDN, DCMDN | https://arxiv.org/abs/1706.02467 |
| *Photometric redshifts for the Kilo-Degree Survey. Machine-learning analysis with artificial neural networks* | NN, BDT | https://arxiv.org/abs/1709.04205 |
| *Estimating Photometric Redshifts for X-ray sources in the X-ATLAS field, using machine-learning techniques* | RF | https://arxiv.org/abs/1710.01313 |
| *Deep learning approach for classifying, detecting and predicting photometric redshifts of quasars in the Sloan Digital Sky Survey stripe 82* | CNN, kNN, SVM, RF, GPR | https://arxiv.org/abs/1712.02777 |
| *Return of the features. Efficient feature selection and interpretation for photometric redshifts* | kNN | https://arxiv.org/abs/1803.10032 |
| *Photometric redshifts from SDSS images using a Convolutional Neural Network* | CNN | https://arxiv.org/abs/1806.06607 | 
| *Estimating redshift distributions using Hierarchical Logistic Gaussian processes* | GPR | https://arxiv.org/abs/1904.09988 |
| *Gaussian Mixture Models for Blended Photometric Redshifts* | GMM | https://arxiv.org/abs/1907.10572 | 
| *Photometric Redshift Calibration with Self Organising Maps* | SOM | https://arxiv.org/abs/1909.09632 |
| *Reliable Photometric Membership (RPM) of Galaxies in Clusters. I. A Machine Learning Method and its Performance in the Local Universe* | SVM | https://arxiv.org/abs/2002.07263 |
| *PhotoWeb redshift: boosting photometric redshift accuracy with large spectroscopic surveys* | CNN | https://arxiv.org/abs/2003.10766 |
| *The PAU Survey: Photometric redshifts using transfer learning from simulations* | MDN | https://arxiv.org/abs/2004.07979 |
| *KiDS+VIKING-450: Improved cosmological parameter constraints from redshift calibration with self-organising maps* | SOM | https://arxiv.org/abs/2005.04207 |
| *Determining the systemic redshift of Lyman-α emitters with neural networks and improving the measured large-scale clustering* | NN | https://arxiv.org/abs/2005.12931 |

<a name='otherobservational'></a>
#### Other Observational
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *Use of neural networks for the identification of new z>=3.6 QSOs from FIRST-SDSS DR5* | NN | https://arxiv.org/abs/0809.0547 | 
| *Estimating the Mass of the Local Group using Machine Learning Applied to Numerical Simulations* | NN | https://arxiv.org/abs/1606.02694 |
| *A probabilistic approach to emission-line galaxy classification* | GMM | https://arxiv.org/abs/1703.07607 |
| *Deep Learning of Quasar Spectra to Discover and Characterize Damped Lya Systems* | CNN | https://arxiv.org/abs/1709.04962 | 
| *An automatic taxonomy of galaxy morphology using unsupervised machine learning* | SOM | https://arxiv.org/abs/1709.05834 | 
| *Learning from the machine: interpreting machine learning algorithms for point- and extended- source classification* | RF, ADA, EXT, BDT, MINT, TINT | https://arxiv.org/abs/1712.03970 | 
| *Predicting the Neutral Hydrogen Content of Galaxies From Optical Data Using Machine Learning* | OLR, RF, BDT, kNN, SVM, NN | https://arxiv.org/abs/1803.08334 |
| *Classifying galaxy spectra at 0.5<z<1 with self-organizing maps* | SOM | https://arxiv.org/abs/1805.07845 |
| *Knowledge transfer of Deep Learning for galaxy morphology from one survey to another* | CNN | https://arxiv.org/abs/1807.00807 |
| *Classification of Broad Absorption Line Quasars with a Convolutional Neural Network* | CNN | https://arxiv.org/abs/1901.04506 | 
| *Generative deep fields: arbitrarily sized, random synthetic astronomical images through deep learning* | GAN | https://arxiv.org/abs/1904.10286 |
| *Deconfusing intensity maps with neural networks* | CNN | https://arxiv.org/abs/1905.10376 |
| *Deep-CEE I: Fishing for Galaxy Clusters with Deep Neural Nets* | RCNN | https://arxiv.org/abs/1906.08784 |
| *Improving Galaxy Clustering Measurements with Deep Learning: analysis of the DECaLS DR7 data* | NN | https://arxiv.org/abs/1907.11355 |
| *What can Machine Learning tell us about the background expansion of the Universe?* | GA | https://arxiv.org/abs/1910.01529 |
| *A deep learning approach to cosmological dark energy models* | BNN+RNN | https://arxiv.org/abs/1910.02788 |
| *Reconstructing Functions and Estimating Parameters with Artificial Neural Network: a test with Hubble parameter and SNe Ia* | NN, RNN, LSTM, GRU | https://arxiv.org/abs/1910.03636 |
| *Multi-wavelength properties of radio and machine-learning identified counterparts to submillimeter sources in S2COSMOS* | SVM, XGBoost | https://arxiv.org/abs/1910.03596 |
| *Machine learning computation of distance modulus for local galaxies* | kNN, BDT, NN | https://arxiv.org/abs/1910.07317 | 
| *MILCANN : A neural network assessed tSZ map for galaxy cluster detection* | NN | https://arxiv.org/abs/1702.00075 |
| *Machine Learning meets the redshift evolution of the CMB Temperature* | GA | https://arxiv.org/abs/2002.12700 | 
| *Inverse Cosmography: testing the effectiveness of cosmographic polynomials using machine learning* | RNN+BNN | https://arxiv.org/abs/2005.02807 |
| *Deblending galaxies with Variational Autoencoders: a joint multi-band, multi-instrument approach* | VAE | https://arxiv.org/abs/2005.12039 |
| *Fully probabilistic quasar continua predictions near Lyman-α with conditional neural spline flows* |NF | https://arxiv.org/abs/2006.00615 | 
| *Artificial intelligence and quasar absorption system modelling; application to fundamental constants at high redshift* | AI | https://arxiv.org/abs/2008.02583 |
| *Deep learning the astrometric signature of dark matter substructure* | CNN | https://arxiv.org/abs/2008.11577 |
| *Beyond the Hubble Sequence -- Exploring Galaxy Morphology with Unsupervised Machine Learning* | VAE | https://arxiv.org/abs/2009.11932 |
| *Deep Learning for Line Intensity Mapping Observations: Information Extraction from Noisy Maps* | GAN | https://arxiv.org/abs/2010.00809 |

&nbsp;
---
<a name='parameters'></a>
## Parameter Estimation
Cosmological parameter estimation is the mechanism of inferring the contents and evolution of our universe from observations. This topic is quite broad, and therefore parameter estimation papers with a focus on an individual experiment/dataset can be found in other sections (e.g. the Reionization and 21cm section). Note this section is unfinished 

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *Bayesian emulator optimisation for cosmology: application to the Lyman-alpha forest* | GP | https://arxiv.org/abs/1812.04631 |
| *Fast likelihood-free cosmology with neural density estimators and active learning* | MDN, NF (MAF) | https://arxiv.org/abs/1903.00007 |
| *Accelerated Bayesian inference using deep learning* | NN | https://arxiv.org/abs/1903.10860 |
| *Cosmic Inference: Constraining Parameters With Observations and Highly Limited Number of Simulations* | GP | https://arxiv.org/abs/1905.07410 | 
| *Euclid-era cosmology for everyone: Neural net assisted MCMC sampling for the joint 3x2 likelihood* | NN | https://arxiv.org/abs/1907.05881 |
| *Parameters Estimation for the Cosmic Microwave Background with Bayesian Neural Networks* | BNNs | https://arxiv.org/abs/1911.08508 |
| *Flow-Based Likelihoods for Non-Gaussian Inference* | NF | https://arxiv.org/abs/2007.05535 | 
| *Nearest Neighbor distributions: new statistical measures for cosmological clustering* | kNN-CDF | https://arxiv.org/abs/2007.13342 |
| *Likelihood-free inference with neural compression of DES SV weak lensing map statistics* | NF | https://arxiv.org/abs/2009.08459 |

&nbsp;

---
<a name='tools'></a>
## Tools
Contained here are some machine learning tools that are specifically designed for the computational challenges of cosmology. 

| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *CosmoFlow: Using Deep Learning to Learn the Universe at Scale* | CNN | https://arxiv.org/abs/1808.04728 |
| *Convolutional Neural Networks on the HEALPix sphere: a pixel-based algorithm and its application to CMB data analysis* | CNN | https://arxiv.org/abs/1902.04083 | 
| *CosmicNet I: Physics-driven implementation of neural networks within Boltzmann-Einstein solvers* | NN | https://arxiv.org/abs/1907.05764 |

&nbsp;

---
<a name='datasets'></a>
## Public Datasets
Contained here are some cosmological machine learning datasets. 

| Title | arxiv link | github link |
| :--- | :---: | :---: |
| *Aemulus Project* | https://arxiv.org/abs/1804.05865 | https://aemulusproject.github.io/ |
| *The Quijote simulations* | https://arxiv.org/abs/1909.05273 | https://github.com/franciscovillaescusa/Quijote-simulations | 
| *The CAMELS project: Cosmology and Astrophysics with MachinE Learning Simulations* | https://arxiv.org/abs/2010.00619 | https://www.camel-simulations.org/ | 
&nbsp;

---
<a name='reviews'></a>
## Reviews
Reviews of machine learning in cosmology, and, more broadly, astronomy.

| Title | arxiv link |
| :---  | :---: |
| *Data Mining and Machine Learning in Astronomy* | https://arxiv.org/abs/0906.2173 | 
| *The Role of Machine Learning in the Next Decade of Cosmology* | https://arxiv.org/abs/1902.10159 |
| *Machine learning and the physical sciences* | https://arxiv.org/abs/1903.10563 |

&nbsp;

---
<a name='acknowledgments'></a>
## Acknowledgments
Thanks to the following people for bringing additional papers to my attention!

Philippe Berger

Dana Simard

Michelle Ntampaka

Farida Farsian

Celia Escamilla-Rivera