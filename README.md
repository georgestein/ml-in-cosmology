## ml-in-cosmology
This github attempts to maintain a comprehensive list of published machine learning applications to cosmology, organized by subject matter and arxiv posting date. I've also included some more obsevational galaxy studies.

Each entry contains the paper title, a simple summary of the machine learning methods used in the work, and the arxiv link. If I have missed any papers that you believe should be included please email me at gstein@berkeley.edu or issue a pull request. 

For general reviews of the subject, or for public machine-learning ready datasets, see the resources at the bottom of this list. 

Feel free to cite in any works
[![DOI](https://zenodo.org/badge/160429652.svg)](https://zenodo.org/badge/latestdoi/160429652)

I am currently a postdoctoral researcher at the Berkeley Center for Cosmological Physics and Lawrence Berkeley National Laboratory, broadly working on machine learning in cosmology. Cheers to whoever can find which of the papers below have me as a (co-)author :beers: 

---
### Table of Contents
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
* SSL: [Self-Supervised Learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
* SVM: [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
* SVR: [Support Vector Regression](https://en.wikipedia.org/wiki/Support_vector_machine)
* TINT: [Tree-Interpreter](https://github.com/andosa/treeinterpreter)
* V-Net: [three dimensional](https://arxiv.org/abs/1606.04797) [U-Net](http://deeplearning.net/tutorial/unet.html)
* XGBoost: [eXtreme Gradient Boosting](https://xgboost.readthedocs.io/en/latest/)

&nbsp;

---
<a name='structure'></a>
## Large-Scale Structure


The [Large-Scale Structure](https://ned.ipac.caltech.edu/level5/March12/Coil/Coil1.html) of the universe is a field that largely relies on state-of-the art cosmological simulations. Due to the computational complexity of these simulations, some investigations will remain computationally-infeasible for the forseeable future, and machine learning techniques can have a number of important uses.

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
| *AI-assisted super-resolution cosmological simulations* | GAN | https://arxiv.org/abs/2010.06608 | 
| *Encoding large scale cosmological structure with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/2011.05244 |
| *Deep learning insights into cosmological structure formation* | CNN |  https://arxiv.org/abs/2011.10577 |
| *SHAPing the Gas: Understanding Gas Shapes in Dark Matter Haloes with Interpretable Machine Learning* | XGBoost | https://arxiv.org/abs/2011.12987 |
| *The BACCO Simulation Project: A baryonification emulator with Neural Networks* | NN | https://arxiv.org/abs/2011.15018 |
| *dm2gal: Mapping Dark Matter to Galaxies with Neural Networks* | CNN | https://arxiv.org/abs/2012.00186 |
| *Fast and Accurate Non-Linear Predictions of Universes with Deep Learning* | V-Net | https://arxiv.org/abs/2012.00240 |
| *Identifying Cosmological Information in a Deep Neural Network* | CNN | https://arxiv.org/abs/2012.03778 |
| *CosmicRIM : Reconstructing Early Universe by Combining Differentiable Simulations with Recurrent Inference Machines* | RIM | https://arxiv.org/abs/2104.12864 |
| *AI-assisted super-resolution cosmological simulations II: Halo substructures, velocities and higher order statistics* | GAN | https://arxiv.org/abs/2105.01016 |
| *Cosmic Velocity Field Reconstruction Using AI* | V-Net | https://arxiv.org/abs/2105.09450 |
| *Normalizing flows for random fields in cosmology* | NF | https://arxiv.org/abs/2105.12024 |
| *Cosmic Voids in GAN-Generated Maps of Large-Scale Structure* | GAN | https://arxiv.org/abs/2106.04014 |
| *Classification algorithms applied to structure formation simulations* | RF | https://arxiv.org/abs/2106.06587 |
| *Fast, high-fidelity Lyman α forests with convolutional neural networks* | V-Net | https://arxiv.org/abs/2106.12662 | 
| *HyPhy: Deep Generative Conditional Posterior Mapping of Hydrodynamical Physics* | VAE | https://arxiv.org/abs/2106.12675 |
| *Predicting halo occupation and galaxy assembly bias with machine learning* | RF | https://arxiv.org/abs/2107.01223 |
| *Finding universal relations in subhalo properties with artificial intelligence* | NN | https://arxiv.org/abs/2109.04484 |
| *Multifield Cosmology with Artificial Intelligence* | CNN | https://arxiv.org/abs/2109.09747 |
| *Robust marginalization of baryonic effects for cosmological inference at the field level* | CNN | https://arxiv.org/abs/2109.10360 |
| *Inpainting hydrodynamical maps with deep learning* | CNN | https://arxiv.org/abs/2109.07070 |
| *Deep learning reconstruction of three-dimensional galaxy distributions with intensity mapping observations* | CNN | https://arxiv.org/abs/2110.05755 | 
| *From EMBER to FIRE: predicting high resolution baryon fields from dark matter simulations with Deep Learning* | GAN | https://arxiv.org/abs/2110.11970 |
| *Modeling the galaxy-halo connection with machine learning* | RF | https://arxiv.org/abs/2111.02422 |
| *Super-resolving Dark Matter Halos using Generative Deep Learning* | U-Net, GAN | https://arxiv.org/abs/2111.06393 |
| *Inferring halo masses with Graph Neural Networks* | GNN | https://arxiv.org/abs/2111.08683 |
| *How to quantify fields or textures? A guide to the scattering transform* | ST | https://arxiv.org/abs/2112.01288 |
| *Neural Network Acceleration of Large-scale Structure Theory Calculations* | NN | https://arxiv.org/abs/2112.05889 | 
| *Physical Benchmarking for AI-Generated Cosmic Web* | U-Net | https://arxiv.org/abs/2112.05681 |
| *Multi-Epoch Machine Learning 1: Unravelling Nature vs Nurture for Galaxy Formation* | ERT | https://arxiv.org/abs/2112.08424 |
| *Unravelling the role of cosmic velocity field in dark matter halo mass function using deep learning* | CNN | https://arxiv.org/abs/2112.14743 |
| *Cosmology with one galaxy?* | NN, GBT | https://arxiv.org/abs/2201.02202 |
| *Mimicking the halo-galaxy connection using machine learning* | DT, kNN, GBM, NN | https://arxiv.org/abs/2201.06054 | 

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
| *Self-supervised Learning with Physics-aware Neural Networks I: Galaxy Model Fitting* | AE | https://arxiv.org/abs/1907.03957 | 
| *Using X-Ray Morphological Parameters to Strengthen Galaxy Cluster Mass Estimates via Machine Learning* | RF | https://arxiv.org/abs/1908.02765 |
| *Large-scale structures in the LCDM Universe: network analysis and machine learning* | XGBoost | https://arxiv.org/abs/1910.07868 |
| *Dynamical mass inference of galaxy clusters with neural flows* | NF (MADE) | https://arxiv.org/abs/2003.05951 |
| *Mass Estimation of Galaxy Clusters with Deep Learning I: Sunyaev-Zel'dovich Effect* | U-Net | https://arxiv.org/abs/2003.06135 |
| *Galaxy cluster mass estimation with deep learning and hydrodynamical simulations* | CNN | https://arxiv.org/abs/2005.11819 |
| *Mass Estimation of Galaxy Clusters with Deep Learning II: CMB Cluster Lensing* | U-NET | https://arxiv.org/abs/2005.13985 |
| *Multiwavelength classification of X-ray selected galaxy cluster candidates using convolutional neural networks* | CNN | https://arxiv.org/abs/2006.05998 |
| *Anomaly detection in Astrophysics: a comparison between unsupervised Deep and Machine Learning on KiDS data* | AE, RF | https://arxiv.org/abs/2006.08235 |
| *Approximate Bayesian Uncertainties on Deep Learning Dynamical Mass Estimates of Galaxy Clusters* | BNN | https://arxiv.org/abs/2006.13231 |
| *A deep learning view of the census of galaxy clusters in IllustrisTNG* | CNN | https://arxiv.org/abs/2007.05144 |
| *Revealing the Local Cosmic Web by Deep Learning* | V-Net | https://arxiv.org/abs/2008.01738 |
| *Simulation-based inference of dynamical galaxy cluster masses with 3D convolutional neural networks* | CNN | https://arxiv.org/abs/2009.03340 |
| *Weak-lensing Mass Reconstruction of Galaxy Clusters with Convolutional Neural Network* | CNN | https://arxiv.org/abs/2102.05403 |
| *DeepSZ: Identification of Sunyaev-Zel'dovich Galaxy Clusters using Deep Learning* | CNN | https://arxiv.org/abs/2102.13123 |
| *DeepMerge II: Building Robust Deep Learning Algorithms for Merging Galaxy Identification Across Domains* | CNN | https://arxiv.org/abs/2103.01373 |
| *Mass Estimation of Planck Galaxy Clusters using Deep Learning* | CNN | https://arxiv.org/abs/2111.01933 |

&nbsp;

---
<a name='reionization21cm'></a>
## Reionization and 21cm
In cosmology, the process of [Reionization](https://en.wikipedia.org/wiki/Reionization) refers to the period when our universe went from the "Dark Ages" - before major star and galaxy formation - to the ionized state we see today. 

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
| *Emulating the Global 21-cm Signal from Cosmic Dawn and Reionization* | NN | https://arxiv.org/abs/1910.06274 |
| *21cm Global Signal Extraction: Extracting the 21cm Global Signal using Artificial Neural Networks* | NN | https://arxiv.org/abs/1911.02580 |
| *A unified framework for 21cm tomography sample generation and parameter inference with Progressively Growing GANs* | GAN | https://arxiv.org/abs/2002.07940 |
| *Beyond the power spectrum - I: recovering H II bubble size distribution from 21 cm power spectrum with artificial neural networks* | NN | https://arxiv.org/abs/2002.08238 |
| *Foreground modelling via Gaussian process regression: an application to HERA data* | GP | https://arxiv.org/abs/2004.06041 |
| *Predicting 21cm-line map from Lyman α emitter distribution with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/2004.09206 |
| *Constraining the Reionization History using Bayesian Normalizing Flows* | NF | https://arxiv.org/abs/2005.07694 | 
| *Deep-Learning Study of the 21cm Differential Brightness Temperature During the Epoch of Reionization* | CNN | https://arxiv.org/abs/2006.06236 |
| *Removing Astrophysics in 21 cm maps with Neural Networks* | CNN | https://arxiv.org/abs/2006.14305 |
| *Deep Forest: Neural Network reconstruction of the Lyman-alpha forest* | NN | https://arxiv.org/abs/2009.10673 |
| *deep21: a Deep Learning Method for 21cm Foreground Removal* | U-Net | https://arxiv.org/abs/2010.15843 | 
| *Analysing the Epoch of Reionization with three-point correlation functions and machine learning techniques* | NN | https://arxiv.org/abs/2011.14157 |
| *Using Artificial Neural Networks to extract the 21-cm Global Signal from the EDGES data* | NN | https://arxiv.org/abs/2012.00028 |
| *Modeling assembly bias with machine learning and symbolic regression* | RF, SR | https://arxiv.org/abs/2012.00111 |
| *Reconstructing Patchy Reionization with Deep Learning* | U-Net | https://arxiv.org/abs/2101.01214 |
| *Deep learning approach for identification of HII regions during reionization in 21-cm observations* | U-Net | https://arxiv.org/abs/2102.06713 |
| *GLOBALEMU: A novel and robust approach for emulating the sky-averaged 21-cm signal from the cosmic dawn and epoch of reionisation* | NN | https://arxiv.org/abs/2104.04336 |
| *Machine learning galaxy properties from 21 cm lightcones: impact of network architectures and signal contamination* | CNN | https://arxiv.org/abs/2107.00018 |
| *21cmVAE: A VAE-based Emulator of the 21-cm Global Signal* | VAE | https://arxiv.org/abs/2107.05581 | 
| *Probing Ultra-light Axion Dark Matter from 21cm Tomography using Convolutional Neural Networks* | CNN | https://arxiv.org/abs/2108.07972 |
| *Deep Forest: Neural Network reconstruction of the Lyman-alpha forest* | NN | https://arxiv.org/abs/2009.10673 |
| *HIFlow: Generating Diverse HI Maps Conditioned on Cosmology using Normalizing Flow* | NF | https://arxiv.org/abs/2110.02983 |
| *Understanding the Impact of Semi-Numeric Reionization Models when using CNNs* | CNN | https://arxiv.org/abs/2112.03443 |
| *Extracting the 21-cm Power Spectrum and the reionization parameters from mock datasets using Artificial Neural Networks* | NN | https://arxiv.org/abs/2112.13866 |
| *Inferring Astrophysics and Dark Matter Properties from 21cm Tomography using Deep Learning* | CNN | https://arxiv.org/abs/2201.07587 |
| *Machine Learning to Decipher the Astrophysical Processes at Cosmic Dawn* | NN | https://arxiv.org/abs/2201.08205 |

&nbsp;
---
<a name='lensing'></a>
## Gravitational Lensing
[Gravitational lensing](http://www.cfhtlens.org/public/what-gravitational-lensing) in cosmology refers to the bending of light due to mass between the source and Earth. This effect is very useful for inferring properties of the total mass distribution in our Universe, which is dominated by dark matter that we cannot see electromagnetically. Gravitational lensing comes in two types: weak and strong.

Strong gravitational lensing refers to the cases where the lensing effect is strong enough to be seen by the human eye, or equivalent, on an astronomical image. For example, we might see multiple images of a background galaxy, or clear shape distortions. This only happens when a massive galaxy cluster lies between us and some background galaxies

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
| *Shear measurement bias II: a fast machine learning calibration method* | MLP | https://arxiv.org/abs/2006.07011 |
| *Interpreting deep learning models for weak lensing* | CNN | https://arxiv.org/abs/2007.06529 |
| *Probabilistic Mapping of Dark Matter by Neural Score Matching* | DE | https://arxiv.org/abs/2011.08271 | 
| *Higher order statistics of shear field: a machine learning approach* | kNN, SVM, GP, RF, etc.. | https://arxiv.org/abs/2011.10438 |
| *Simultaneously constraining cosmology and baryonic physics via deep learning from weak lensing* | CNN | https://arxiv.org/abs/2109.11060 |
| *Probabilistic Mass Mapping with Neural Score Estimation* | NSM | https://arxiv.org/abs/2201.05561 |

<a name='strong'></a>
#### Strong Lensing
| Title | ML technique(s) used | arxiv link |
| :--- | :---: | :---: |
| *A neural network gravitational arc finder based on the Mediatrix filamentation method* | NN | https://arxiv.org/abs/1607.04644 |
| *CMU DeepLens: deep learning for automatic image-based galaxy-galaxy strong lens finding* | CNN | https://arxiv.org/abs/1703.02642 |
| *Automated Lensing Learner: Automated Strong Lensing Identification with a Computer Vision Technique* | HoG | https://arxiv.org/abs/1704.02322 |
| *Finding strong lenses in CFHTLS using convolutional neural networks* | CNN | https://arxiv.org/abs/1704.02744 |
| *Fast automated analysis of strong gravitational lenses with convolutional neural networks* | CNN | https://arxiv.org/abs/1708.08842 |
| *Uncertainties in Parameters Estimated with Neural Networks: Application to Strong Gravitational Lensing* | NN | https://arxiv.org/abs/1708.08843 |
| *The Strong Gravitational Lens Finding Challenge* | SVM, CNN | https://arxiv.org/abs/1802.03609 | 
| *Testing convolutional neural networks for finding strong gravitational lenses in KiDS* | CNN | https://arxiv.org/abs/1807.04764 |
| *Analyzing interferometric observations of strong gravitational lenses with recurrent and convolutional neural networks* | RNN, CNN | https://arxiv.org/abs/1808.00011 |
| *Data-Driven Reconstruction of Gravitationally Lensed Galaxies using Recurrent Inference Machines* | RIM, CNN | https://arxiv.org/abs/1901.01359 |
| *Finding Strong Gravitational Lenses in the DESI DECam Legacy Survey* | CNN | https://arxiv.org/abs/1906.00970 |
| *Mining for Dark Matter Substructure: Inferring subhalo population properties from strong lenses with machine learning* | NN | https://arxiv.org/abs/1909.02005 |
| *Deep Learning the Morphology of Dark Matter Substructure* | CNN | https://arxiv.org/abs/1909.07346 |
| *Circumventing Lens Modeling to Detect Dark Matter Substructure in Strong Lens Images with Convolutional Neural Networks* | CNN | https://arxiv.org/abs/1910.00015 |
| *Differentiable Strong Lensing: Uniting Gravity and Neural Nets through Differentiable Probabilistic Programming* | VAE | https://arxiv.org/abs/1910.06157 |
| *Identifying Strong Lenses with Unsupervised Machine Learning using Convolutional Autoencoder* | VAE | https://arxiv.org/abs/1911.04320 || *Modular Deep Learning Analysis of Galaxy-Scale Strong Lensing Images* | CNN | https://arxiv.org/abs/1911.03867 | 
| *HOLISMOKES II. Identifying galaxy-scale strong gravitational lenses in Pan-STARRS using convolutional neural networks* | CNN | https://arxiv.org/abs/2004.13048 | 
| *Discovering New Strong Gravitational Lenses in the DESI Legacy Imaging Surveys* | CNN | https://arxiv.org/abs/2005.04730 |
| *Dark Matter Subhalos, Strong Lensing and Machine Learning* | CNN | https://arxiv.org/abs/2005.05353 |
| *Deep Learning for Strong Lensing Search: Tests of the Convolutional Neural Networks and New Candidates from KiDS DR3* | CNN | https://arxiv.org/abs/2007.00188 |
| *Decoding Dark Matter Substructure without Supervision* | AE, VAE, AAE | https://arxiv.org/abs/2008.12731 |
| *Extracting the Subhalo Mass Function from Strong Lens Images with Image Segmentation* | U-Net | https://arxiv.org/abs/2009.06639 | 
| *Detecting Subhalos in Strong Gravitational Lens Images with Image Segmentation* | U-Net | https://arxiv.org/abs/2009.06663 |
| *Hunting for Dark Matter Subhalos in Strong Gravitational Lensing with Neural Networks* | CNN | https://arxiv.org/abs/2010.12960 |
| *Targeted Likelihood-Free Inference of Dark Matter Substructure in Strongly-Lensed Galaxies* | GP, ... | https://arxiv.org/abs/2010.07032 | 
| *Large-Scale Gravitational Lens Modeling with Bayesian Neural Networks for Accurate and Precise Inference of the Hubble Constant* | BNN | https://arxiv.org/abs/2012.00042 |
| *Strong lens systems search in the Dark Energy Survey using Convolutional Neural Networks* | CNN | https://arxiv.org/abs/2109.00014 |
| *Finding quadruply imaged quasars with machine learning. I. Methods* | CNN, VAE | https://arxiv.org/abs/2109.09781 |
| *Mining for strong gravitational lenses with self-supervised learning* | SSL | https://arxiv.org/abs/2110.00023 |
| *High-quality strong lens candidates in the final Kilo Degree survey footprint* | CNN | https://arxiv.org/abs/2110.01905 |
| *The DES Bright Arcs Survey: Candidate Strongly Lensed Galaxy Systems from the Dark Energy Survey 5,000 Sq. Deg. Footprint* | CNN | https://arxiv.org/abs/2110.02418 |
| *Finding Strong Gravitational Lenses Through Self-Attention* | CNN, ViT | https://arxiv.org/abs/2110.09202 |
| *A search for galaxy-scale strong gravitational lenses in the Ultraviolet Near Infrared Optical Northern Survey (UNIONS)* | CNN | https://arxiv.org/abs/2110.11972 |
| *Exploring the interpretability of deep neural networks used for gravitational lens finding with a sensitivity probe* | CNN | https://arxiv.org/abs/2112.02479 |
| *Detecting gravitational lenses using machine learning: exploring interpretability and sensitivity to rare lensing configurations* | CNN | https://arxiv.org/abs/2202.12776 |

&nbsp;

---
<a name='cmb'></a>
## Cosmic Microwave Background
The [Cosmic Microwave Background (CMB)](https://en.wikipedia.org/wiki/Cosmic_microwave_background) is the light left over from the period of recombination in the very early Universe, 380,000 years after the beginning. CMB observations are sometimes referred to as "baby pictures of our Universe", as this light has been travelling for 13.5 billion years just to reach us. As this light travels towards us it can be effected by the large scale structure along the line of sight, which can be used as a probe of structure formation.

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
| *Inpainting CMB maps using Partial Convolutional Neural Networks* | U-Net | https://arxiv.org/abs/2011.01433 |
| *ForSE: a GAN based algorithm for extending CMB foreground models to sub-degree angular scales* | GAN | https://arxiv.org/abs/2011.02221 |
| *A Generative Model of Galactic Dust Emission Using Variational Inference* | VAE | https://arxiv.org/abs/2101.11181 |
| *A convolutional-neural-network estimator of CMB constraints on dark matter energy injection* | CNN | https://arxiv.org/abs/2101.10360 |
| *An Unbiased Estimator of the Full-sky CMB Angular Power Spectrum using Neural Networks* | NN | https://arxiv.org/abs/2102.04327 |
| *MillimeterDL: Deep Learning Simulations of the Microwave Sky* | U-Net | https://arxiv.org/abs/2105.11444 |
| *Reconstructing Cosmic Polarization Rotation with ResUNet-CMB* | U-Net | https://arxiv.org/abs/2109.09715 |
| *Single frequency CMB B-mode inference with realistic foregrounds from a single training image* | U-Net | https://arxiv.org/abs/2111.01138 |
| *Convolutional Neural Network-reconstructed velocity for kinetic SZ detection* | CNN | https://arxiv.org/abs/2201.01643 |
| *Cosmic Kite: Auto-encoding the Cosmic Microwave Background* | AE | https://arxiv.org/abs/2202.05853 |

&nbsp;

---
<a name='observational'></a>
## Observational

This section has a variety of machine learning papers used for various observational applications, mainly focusing on redshift estimation and various galaxy classification problems.

<a name='redshifts'></a>
#### Redshifts

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
| *PS1-STRM: Neural network source classification and photometric redshift catalogue for PS1* | NN | https://arxiv.org/abs/1910.10167 |
| *Reliable Photometric Membership (RPM) of Galaxies in Clusters. I. A Machine Learning Method and its Performance in the Local Universe* | SVM | https://arxiv.org/abs/2002.07263 |
| *PhotoWeb redshift: boosting photometric redshift accuracy with large spectroscopic surveys* | CNN | https://arxiv.org/abs/2003.10766 |
| *The PAU Survey: Photometric redshifts using transfer learning from simulations* | MDN | https://arxiv.org/abs/2004.07979 |
| *KiDS+VIKING-450: Improved cosmological parameter constraints from redshift calibration with self-organising maps* | SOM | https://arxiv.org/abs/2005.04207 |
| *Determining the systemic redshift of Lyman-α emitters with neural networks and improving the measured large-scale clustering* | NN | https://arxiv.org/abs/2005.12931 |
| *Photometric selection and redshifts for quasars in the Kilo-Degree Survey Data Release 4* | RF, XGBoost, NN | https://arxiv.org/abs/2010.13857 |  
| *Photometric Redshift Estimation with a Convolutional Neural Network: NetZ* | CNN | https://arxiv.org/abs/2011.12312 |
| *A machine learning approach to galaxy properties: joint redshift-stellar mass probability distributions with Random Forest* | RF | https://arxiv.org/abs/2012.05928 |
| *Spectroscopic and Photometric Redshift Estimation by Neural Networks For the China Space Station Optical Survey (CSS-OS)* | NN | https://arxiv.org/abs/2101.02532 |
| *Estimating Galactic Distances From Images Using Self-supervised Representation Learning* | SSL | https://arxiv.org/abs/2101.04293 |
| *QSO photometric redshifts using machine learning and neural networks* | kNN, DT, NN | https://arxiv.org/abs/2102.09177 | 
| *Benchmarking and Scalability of Machine Learning Methods for Photometric Redshift Estimation* | RF, BDT, kNN | https://arxiv.org/abs/2104.01875 |
| *Z-Sequence: Photometric redshift predictions for galaxy clusters with sequential random k-nearest neighbours* | kNN | https://arxiv.org/abs/2104.11335 |
| *Probabilistic photo-z machine learning models for X-ray sky surveys* | RF | https://arxiv.org/abs/2107.01891 |
| *Non-Sequential Neural Network for Simultaneous, Consistent Classification and Photometric Redshifts of OTELO Galaxies* | NN | https://arxiv.org/abs/2108.09415 |
| *Using a Neural Network Classifier to Select Galaxies with the Most Accurate Photometric Redshifts* | NN | https://arxiv.org/abs/2108.13260 |
| *Investigating Deep Learning Methods for Obtaining Photometric Redshift Estimations from Images* | RF, CNN | https://arxiv.org/abs/2109.02503 |
| *Estimation of Photometric Redshifts. I. Machine Learning Inference for Pan-STARRS1 Galaxies Using Neural Networks* | NN | https://arxiv.org/abs/2110.05726 | 
| *Photometric Redshift Estimation of BASS DR3 Quasars by Machine Learning* | RF, XGBoost | https://arxiv.org/abs/2110.14951 |
| *Photometric redshifts for the S-PLUS Survey: is machine learning up to the task?* | GP, NN, MDN | https://arxiv.org/abs/2110.13901 |
| *Machine learning synthetic spectra for probabilistic redshift estimation: SYTH-Z* | MDN | https://arxiv.org/abs/2111.12118 |
| *Photometric Redshifts from SDSS Images with an Interpretable Deep Capsule Network* | CNN | https://arxiv.org/abs/2112.03939 |
| *Estimation of Photometric Redshifts. II. Identification of Out-of-Distribution Data with Neural Networks*| NN | https://arxiv.org/abs/2112.07104 |
| *Machine Learning Classification to Identify Catastrophic Outlier Photometric Redshift Estimates* | NN | https://arxiv.org/abs/2112.07811 |
| *Extracting Photometric Redshift from Galaxy Flux and Image Data using Neural Networks in the CSST Survey* | NN | https://arxiv.org/abs/2112.08690 |
| *Photometric Redshifts for Cosmology: Improving Accuracy and Uncertainty Estimates Using Bayesian Neural Networks* | BNN | https://arxiv.org/abs/2202.07121 |
| *The sensitivity of GPz estimates of photo-z posterior PDFs to realistically complex training set imperfections* | GP | https://arxiv.org/abs/2202.12775 |
| *Photometric Redshift Estimation with Convolutional Neural Networks and Galaxy Images: A Case Study of Resolving Biases in Data-Driven Methods* | CNN | https://arxiv.org/abs/2202.09964 |

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
| *Star-galaxy classification in the Dark Energy Survey Y1 dataset* | SVM, ADA | https://arxiv.org/abs/1805.02427 |
| *Classifying galaxy spectra at 0.5<z<1 with self-organizing maps* | SOM | https://arxiv.org/abs/1805.07845 |
| *Radio Galaxy Zoo: ClaRAN - A Deep Learning Classifier for Radio Morphologies* | CNN | https://arxiv.org/abs/1805.12008 |
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
| *Fully probabilistic quasar continua predictions near Lyman-α with conditional neural spline flows* | NF | https://arxiv.org/abs/2006.00615 | 
| *Artificial intelligence and quasar absorption system modelling; application to fundamental constants at high redshift* | AI | https://arxiv.org/abs/2008.02583 |
| *Deep learning the astrometric signature of dark matter substructure* | CNN | https://arxiv.org/abs/2008.11577 |
| *Beyond the Hubble Sequence -- Exploring Galaxy Morphology with Unsupervised Machine Learning* | VAE | https://arxiv.org/abs/2009.11932 |
| *Deep Learning for Line Intensity Mapping Observations: Information Extraction from Noisy Maps* | GAN | https://arxiv.org/abs/2010.00809 |
| *Peculiar Velocity Estimation from Kinetic SZ Effect using Deep Neural Networks* | CNN | https://arxiv.org/abs/2010.03762 |
| *Machine learning forecasts of the cosmic distance duality relation with strongly lensed gravitational wave events* | GP, GA | https://arxiv.org/abs/2011.02718 |
| *Survey2Survey: A deep learning generative model approach for cross-survey image mapping* | AE, GAN | https://arxiv.org/abs/2011.07124 |
| *DeepShadows: Separating Low Surface Brightness Galaxies from Artifacts using Deep Learning* | CNN | https://arxiv.org/abs/2011.12437 |
| *Model independent calibrations of gamma ray bursts using machine learning* | RF, NN | https://arxiv.org/abs/2011.13590 |
| *Self-Supervised Representation Learning for Astronomical Images* | SSL | https://arxiv.org/abs/2012.13083 |
| *An Active Galactic Nucleus Recognition Model based on Deep Neural Network* | NN | https://arxiv.org/abs/2101.06683 |
| *A Machine Learning Approach to Measuring the Quenched Fraction of Low-Mass Satellites Beyond the Local Group* | NN | https://arxiv.org/abs/2102.05050 |
| *The PAU survey: Estimating galaxy photometry with deep learning* | CNN | https://arxiv.org/abs/2104.02778 |
| *Capturing the physics of MaNGA galaxies with self-supervised Machine Learning* | SSL | https://arxiv.org/abs/2104.08292 |
| *Anomaly detection in Hyper Suprime-Cam galaxy images with generative adversarial networks* | GAN, AE | https://arxiv.org/abs/2105.02434 | 
| *Euclid preparation: XVI. Forecasts for galaxy morphology with the Euclid Survey using Deep Generative Models* | VAE | https://arxiv.org/abs/2105.12149 |
| *Planck Limits on Cosmic String Tension Using Machine Learning* | CNN | https://arxiv.org/abs/2106.00059 |
| *Morphological classification of compact and extended radio galaxies using convolutional neural networks and data augmentation techniques* | CNN | https://arxiv.org/abs/2107.00385 |
| *Galaxy Deblending using Residual Dense Neural networks* | RDN | https://arxiv.org/abs/2109.09550 |
| *An astronomical image content-based recommendation system using combined deep learning models in a fully unsupervised mode* | SOM | https://arxiv.org/abs/2103.00276 |
| *Emulating Sunyaev-Zeldovich Images of Galaxy Clusters using Auto-Encoders* | AE | https://arxiv.org/abs/2110.02232 |
| *Deep Transfer Learning for Blended Source Identification in Galaxy Survey Data* | CNN | https://arxiv.org/abs/2110.08180 |
| *Explaining deep learning of galaxy morphology with saliency mapping* | CNN | https://arxiv.org/abs/2110.08288 |
| *SDSS-IV DR17: Final Release of MaNGA PyMorph Photometric and Deep Learning Morphological Catalogs* | CNN | https://arxiv.org/abs/2110.10694 |
| *Realistic galaxy image simulation via score-based generative models* | DDPM | https://arxiv.org/abs/2111.01713 |
| *A Comparison of Deep Learning Architectures for Optical Galaxy Morphology Classification* | CNN | https://arxiv.org/abs/2111.04353 |
| *Weight Pruning and Uncertainty in Radio Galaxy Classification* | CNN | https://arxiv.org/abs/2111.11654 | 
| *Probabilistic segmentation of overlapping galaxies for large cosmological surveys* | U-Net | https://arxiv.org/abs/2111.15455 |
| *Radio Galaxy Zoo: Giant Radio Galaxy Classification using Multi-Domain Deep Learning* | CNN | https://arxiv.org/abs/2112.03564 |
| *Deep Learning of DESI Mock Spectra to Find Damped Lyα Systems* | CNN | https://arxiv.org/abs/2201.00827 |
| *Astronomical Image Colorization and upscaling with Generative Adversarial Networks* | GAN | https://arxiv.org/abs/2112.13865 |
| *Partial-Attribution Instance Segmentation for Astronomical Source Detection and Deblending* | CNN | https://arxiv.org/abs/2201.04714 |
| *A Hitchhiker's Guide to Anomaly Detection with Astronomaly* | CNN, NN | https://arxiv.org/abs/2201.10189 |
| *Classifying Galaxy Morphologies with Few-Shot Learning* | CNN | https://arxiv.org/abs/2202.08172 |

&nbsp;
---
<a name='parameters'></a>
## Parameter Estimation
Cosmological parameter estimation is the mechanism of inferring the contents and evolution of our universe from observations. This topic is quite broad, and therefore parameter estimation papers with a focus on an individual experiment/dataset can be found in other sections (e.g. the Reionization and 21cm section). Note that there is an abundance of work in this area that is not covered in the few examples below.  

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
| *Neural networks as optimal estimators to marginalize over baryonic effects* | NN | https://arxiv.org/abs/2011.05992 |
| *Solving high-dimensional parameter inference: marginal posterior densities & Moment Networks* | NF | https://arxiv.org/abs/2011.05991 |
| *Accelerating MCMC algorithms through Bayesian Deep Networks* | BNN | https://arxiv.org/abs/2011.14276 |
| *Seeking New Physics in Cosmology with Bayesian Neural Networks I: Dark Energy and Modified Gravity* | BNN | https://arxiv.org/abs/2012.03992 |
| *Unsupervised Resource Allocation with Graph Neural Networks* | GNN | https://arxiv.org/abs/2106.09761 |
| *Machine-driven searches for cosmological physics* | IM | https://arxiv.org/abs/2107.00657 |
| *Lossless, Scalable Implicit Likelihood Inference for Cosmological Fields* | NN | https://arxiv.org/abs/2107.07405 |
| *Translation and Rotation Equivariant Normalizing Flow (TRENF) for Optimal Cosmological Analysis* | NF | https://arxiv.org/abs/2202.05282 |
 
&nbsp;

---
<a name='tools'></a>
## Tools
Contained here are some machine learning tools that are specifically designed for the computational challenges of cosmology. 

| Title | arxiv link |
| :--- | :---: |
| *CosmoFlow: Using Deep Learning to Learn the Universe at Scale* | https://arxiv.org/abs/1808.04728 |
| *DeepSphere: Efficient spherical Convolutional Neural Network with HEALPix sampling for cosmological applications*  | https://arxiv.org/abs/1810.12186 |
| *Convolutional Neural Networks on the HEALPix sphere: a pixel-based algorithm and its application to CMB data analysis*  | https://arxiv.org/abs/1902.04083 | 
| *CosmicNet I: Physics-driven implementation of neural networks within Boltzmann-Einstein solvers* | https://arxiv.org/abs/1907.05764 |
| *FlowPM: Distributed TensorFlow Implementation of the FastPM Cosmological N-body Solver* | https://arxiv.org/abs/2010.11847 |
| *Scattering Networks on the Sphere for Scalable and Rotationally Equivariant Spherical CNNs* | https://arxiv.org/abs/2102.02828 |
| *Towards Machine Learning-Based Meta-Studies: Applications to Cosmological Parameters* | https://arxiv.org/abs/2107.00665 |
| *Equivariant Networks for Pixelized Spheres* | https://arxiv.org/abs/2106.06662 |
&nbsp;

---
<a name='datasets'></a>
## Public Datasets
Contained here are some cosmological datasets geared towards straightforward use on machine learning applications. 

| Title | arxiv link | github link |
| :--- | :---: | :---: |
| *Aemulus Project* | https://arxiv.org/abs/1804.05865 | https://aemulusproject.github.io/ |
| *The Quijote simulations* | https://arxiv.org/abs/1909.05273 | https://github.com/franciscovillaescusa/Quijote-simulations | 
| *The CAMELS project: Cosmology and Astrophysics with MachinE Learning Simulations* | https://arxiv.org/abs/2010.00619 | https://www.camel-simulations.org/ | 
| *The CAMELS Multifield Dataset: Learning the Universe's Fundamental Parameters with Artificial Intelligence* | https://arxiv.org/abs/2109.10915 |

&nbsp;

---
<a name='reviews'></a>
## Reviews
Reviews of machine learning in cosmology, and, more broadly, machine learning in astronomy.

| Title | arxiv link |
| :---  | :---: |
| *Data Mining and Machine Learning in Astronomy* | https://arxiv.org/abs/0906.2173 | 
| *The Role of Machine Learning in the Next Decade of Cosmology* | https://arxiv.org/abs/1902.10159 |
| *Machine learning and the physical sciences* | https://arxiv.org/abs/1903.10563 |
| *Building Trustworthy Machine Learning Models for Astronomy* | https://arxiv.org/abs/2111.14566 | 

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

Michaël Defferrard

Farida Farsian

Pranath Reddy

Camille Avestruz

Harry Bevins

T. Lucas Makinen

Francois Lanusse