## SUPPLEMENTAL MATERIAL

## Sequential Bayesian experiment design for optically detected magnetic resonance of nitrogen-vacancy centers

Sergey Dushenko ${ }^{1,2}$, Kapildeb Ambal $^{1-3}$, Robert D. McMichael ${ }^{1}$<br>${ }^{1}$ Physical Measurement Laboratory, National Institute of Standards and Technology, Gaithersburg, MD 20899, USA<br>${ }^{2}$ Institute for Research in Electronics and Applied Physics, University of Maryland, College Park, MD 20742, USA<br>${ }^{3}$ Department of Physics, Wichita State University, Wichita, KS 67260, USA

## CONTENT

## S. 1 ADDITIONAL DETAILS ON STRUCTURE AND PHYSICS OF NV CENTER

## S. 2 INTRODUCTION TO SEQUENTIAL BAYESIAN EXPERIMENT DESIGN

S.2.1 Bayes' Theorem
S.2.2 Bayesian Inference
S.2.3 Bayesian Experiment design
S.2.4 Evaluation of Utility Function
S. 3 IMPLEMENTATION OF PROBABILITY DISTRIBUTIONS
S. 4 SPECIFICATIONS OF THE COMPUTATIONAL HARDWARE USED FOR SEQUENTIAL BAYESIAN EXPERIMENT DESIGN
S. 5 SPEEDUP OF THE SEQUENTIAL BAYESIAN EXPERIMENT DESIGN

REFERENCES

## S. 1 ADDITIONAL DETAILS ON STRUCTURE AND PHYSICS OF NV CENTER

The NV center is a quantum defect that is created when two adjacent carbon atoms in a diamond lattice are substituted with a vacancy and a negatively charged nitrogen atom. This nitrogenvacancy system in the diamond lattice has six electrons (five electrons from the nitrogen atom and surrounding carbon atoms, plus one additional electron from the lattice), four of which are located on the energy levels inside the energy bandgap of diamond. In the ground and photoexcited states of the NV center, two electrons are unpaired, resulting in a total spin of the system adding up to 1 , and creating a spin $S=1$ quantum defect. NV center can be excited using photons in the wavelength range of 480 nm to 637 nm [1], due to $2.6 \mathrm{eV}(477 \mathrm{~nm})$ energy difference between the ground state and the bottom of the conduction band and a 1.95 eV energy difference between the ground and excited states (637 nm zero-phonon line) [2]. Figure S1 shows the energy level structure of the NV- center. The NVcenter is not susceptible to photobleaching.

![](https://cdn.mathpix.com/cropped/9215236f-3911-4480-ae02-66abbed02411-2.jpg?height=1011&width=1111&top_left_y=981&top_left_x=249)
Fig. S1. Schematic structure of the energy levels of NV center (not to scale). NV center in the ground state can be excited by the green laser (green arrow); the process preserves spin projection. From the excited state NV center can relax back to the ground state by emitting a red photon ( $m_{\mathrm{s}}= \pm 1$ or $m_{\mathrm{s}}=0$ excited states), or non-radiatively relaxing through the dark state (only $m_{\mathrm{s}}= \pm 1$ excited states). Transition between the states with $m_{\mathrm{s}}= \pm 1$ and $m_{\mathrm{s}}=0$ can be induced by the microwaves (blue arrows).

## Valence band

## S. 2 INTRODUCTION TO SEQUENTIAL BAYESIAN EXPERIMENT DESIGN

Sequential design divides measurement runs into a sequence of design-measure-analyze cycles, providing each design decision with information gleaned from all the data that have accumulated up to that point (Fig. S2). In contrast, the traditional method involves a preprogrammed series of settings as the design, a period of measurement, and finally analysis that yields useful information at the very end of the process.

Here, we set the data collection aside and focus on the design and analysis processes in sequential Bayesian experiment design. The methods outlined here have been described previously by numerous authors [3,4]. After a brief introduction to Bayes' theorem (section S.2.1), we describe the application of Bayes' theorem to the analysis process (section S.2.2), and then follow with the use of Bayes' theorem and information theory in making design decisions (sections S.2.3 and S.2.4).

## S.2.1 Bayes' Theorem

The ideas of Bayesian methodology were first proposed by Bayes [5] and then independently rediscovered and developed much further by Laplace [6,7]. Bayes' theorem is built upon concepts of probability distributions like $p(A)$, describing the probability of $A$, conditional probabilities like $p(A \mid B)$ describing the probability of $A$ given $B$, and joint probabilities like $p(A, B)$ describing the probability of both $A$ and $B$. Bayes' theorem follows from the fact that the joint probability can be expressed in terms of a conditional probability: $p(A, B)=p(A \mid B) \cdot p(B)$, the joint probability of both $A$ and $B$ is the conditional probability of $A$ given $B$ times the probability of $B$. But it is equally true that, $p(A, B)= p(B \mid A) \cdot p(A)$. The combination of these last two equations yields Bayes' theorem:

$$
\begin{equation*}
p(A \mid B)=\frac{p(B \mid A)}{p(B)} \cdot p(A) . \tag{S1}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/9215236f-3911-4480-ae02-66abbed02411-3.jpg?height=633&width=1744&top_left_y=1632&top_left_x=244)
FIG. S2. (a) Schematic layout of the NV magnetometry experiment using sequential Bayesian experiment design. (b) Block diagram of the sequential Bayesian experiment design algorithm. UFL stands for user's favorite language.

## S.2.2 Bayesian Inference

In the present context, we are interested in the probability distribution $p(\boldsymbol{\theta})$ over parameters $\boldsymbol{\theta}=\left\{\theta_{1}, \theta_{2}, \ldots, \theta_{k}\right\}$ of a model function $f(\boldsymbol{\theta}, \boldsymbol{d})$ that also depends on experimental settings (designs) $\boldsymbol{d}$. Thinking about model parameters as random variables with a distribution to be determined may seem very different from thinking about the same parameters as fixed numbers with values to be determined and uncertainty due to noise. However, the familiar notation $x=\bar{x} \pm \sigma$ is shorthand notation for a Gaussian distribution. Using Bayes theorem to determine the distribution of parameters $\boldsymbol{\theta}$ ( $A$ in [S1]) given measurement data $\boldsymbol{y}_{n}=\left\{y_{1}, y_{2}, \ldots, y_{n}\right\}$ ( $B$ in [S1]) accumulated after $n$ measurements made using settings $\boldsymbol{d}_{n}=\left\{d_{1}, d_{2}, \ldots, d_{n}\right\}$, substitution yields

$$
\begin{equation*}
p_{n}(\boldsymbol{\theta}) \equiv p\left(\boldsymbol{\theta} \mid \boldsymbol{y}_{n}, \boldsymbol{d}_{n}\right)=\frac{p\left(\boldsymbol{y}_{n} \mid \boldsymbol{\theta}, \boldsymbol{d}_{n}\right)}{p\left(\boldsymbol{y}_{n} \mid \boldsymbol{d}_{n}\right)} p_{0}(\boldsymbol{\theta}) \tag{S2}
\end{equation*}
$$

In Bayesian lingo, $p_{0}(\boldsymbol{\theta})$ is the prior, the distribution of parameter values before measurement data are considered. The posterior, $p\left(\boldsymbol{\theta} \mid \boldsymbol{y}_{n}, \boldsymbol{d}_{n}\right)$ is the parameter distribution given the collected data. The numerator is called the likelihood, and the denominator is the evidence.

With each additional measurement, the parameter distribution can be refined using Bayes' theorem. With result $y_{n+1}$ measured using settings $d_{n+1}$,

$$
\begin{equation*}
p_{n+1}(\boldsymbol{\theta}) \equiv p\left(\boldsymbol{\theta} \mid y_{n+1}, d_{n+1}, \boldsymbol{y}_{n}, \boldsymbol{d}_{n}\right)=\frac{p\left(y_{n+1} \mid \boldsymbol{\theta}, d_{n+1}\right)}{p\left(y_{n+1} \mid d_{n+1}\right)} p\left(\boldsymbol{\theta} \mid \boldsymbol{y}_{n}, \boldsymbol{d}_{n}\right) \tag{S3}
\end{equation*}
$$

In the numerator, the likelihood is a function of the parameter variables $\boldsymbol{\theta}$ with constants $y_{n+1}$ and $d_{n+1}$. It is the probability of getting a measurement result $y_{n+1}$ as a function of $\boldsymbol{\theta}$ when setting design $d_{n+1}$ is used. In the denominator, the evidence $p\left(y_{n+1} \mid d_{n+1}\right)$ is a constant that maintains normalization $\int p(\boldsymbol{\theta}) d \boldsymbol{\theta}=1$.

To estimate the likelihood function, we must provide a connection between settings, parameters and measurement results. Here that connection is provided by a model function $y= f(\boldsymbol{\theta}, \boldsymbol{d})+\eta$ where $\eta$ is a model of experimental noise. The model function is roughly equivalent to the fitting function one would use for least-squares regression.

If the noise $\eta$ follows a normal (Gaussian) distribution with standard deviation $\sigma$, the probability of a measurement yielding $y_{n+1}$ depends on the difference between the measured value and the modeled values as a function of $\boldsymbol{\theta}$ :

$$
\begin{equation*}
p\left(y_{n+1} \mid \boldsymbol{\theta}, d_{n+1}\right)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left[\frac{-\left[y_{n+1}-f\left(\boldsymbol{\theta}, d_{n+1}\right)\right]^{2}}{2 \sigma^{2}}\right] \tag{S4}
\end{equation*}
$$

Qualitatively, some parameter values, say $\boldsymbol{\theta}_{a}$, will produce model results $f\left(\theta_{a}, d_{n+1}\right)$ that are closer to $y_{n+1}$ than will other parameter values $\boldsymbol{\theta}_{\boldsymbol{b}}$. It follows that the likelihood given in [S4] is greater for $\boldsymbol{\theta}_{a}$ than for $\boldsymbol{\theta}_{b}$. In a quantitative way, the likelihood formalizes the notion that $\boldsymbol{\theta}_{a}$ "explains the data" better than $\boldsymbol{\theta}_{b}$. Although the model function does not depend on the noise parameter $\sigma$, the likelihood does. If $\sigma$ is treated as an additional parameter, some values of $\sigma$ will "explain the data" better than others.

## S.2.3 Bayesian Experiment design

We now turn to the problem of selecting a design (choosing settings) for a future measurement, preferably making good use of the refined parameter distribution $p_{n}(\boldsymbol{\theta})$. We will frame the problem in terms of defining a utility function $U(\boldsymbol{d})$ that expresses the predicted benefit of a future measurement made with setting $\boldsymbol{d}$.

First, we look at predicted measurement values and their distributions. The distribution of predicted measurement values for a design $\boldsymbol{d}$ and fixed $\boldsymbol{\theta}$ is

$$
\begin{equation*}
p(y \mid \boldsymbol{\theta}, \boldsymbol{d})=p_{\eta}(y-f(\boldsymbol{\theta}, \boldsymbol{d})) \tag{S5}
\end{equation*}
$$

Where $p_{\eta}(\cdot)$ is the distribution of measurement noise values. To obtain the full distribution of predicted $y$ values, $p(y \mid \boldsymbol{d})$, we must also account for the probability distribution of $\boldsymbol{\theta}$ values by integrating over the $\boldsymbol{\theta}$ values weighted by the $p_{n}(\boldsymbol{\theta})$ distribution.

$$
\begin{equation*}
p(y \mid \boldsymbol{d})=\int p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) p_{n}(\theta) d \theta \tag{S6}
\end{equation*}
$$

Next, in order to make decisions about future measurements, we need to quantify the "goodness" of a $\theta$ distribution. For this purpose, the information entropy is the conventional measure. For an arbitrary distribution $p(x)$ the information entropy is defined as

$$
\begin{equation*}
H=-\int p(x) \ln [p(x)] d x \tag{S7}
\end{equation*}
$$

The change in information entropy of the parameter distribution that would result from a future measurement value $y$ is given by the difference in entropy between the posterior distribution given predicted measurements, $p(\boldsymbol{\theta} \mid y, \boldsymbol{d})$ and the prior distribution $p(\boldsymbol{\theta})$

$$
\begin{equation*}
\Delta H(y \mid \boldsymbol{d})=-\int p(\boldsymbol{\theta} \mid y, \boldsymbol{d}) \ln [p(\boldsymbol{\theta} \mid y, \boldsymbol{d})] d \boldsymbol{\theta}+\int p(\boldsymbol{\theta}) \ln [p(\boldsymbol{\theta})] d \boldsymbol{\theta} \tag{S8}
\end{equation*}
$$

The expectation value of $\Delta H$ is our utility function,

$$
\begin{equation*}
U(\boldsymbol{d})=\int d y p(y \mid \boldsymbol{d}) \Delta H(y \mid \boldsymbol{d}) \tag{S9}
\end{equation*}
$$

which predicts the mean benefit of a future measurement made using setting $\boldsymbol{d}$.
Combining [S8] and [S9], and using Bayes' theorem:

$$
\begin{equation*}
\mathrm{U}(\boldsymbol{d})=-\iint p(y \mid \boldsymbol{d}) \frac{p(y \mid \boldsymbol{\theta}, \boldsymbol{d})}{p(y \mid \boldsymbol{d})} \mathrm{p}(\boldsymbol{\theta}) \ln \left[\frac{p(y \mid \boldsymbol{\theta}, \boldsymbol{d})}{p(y \mid \boldsymbol{d})} \mathrm{p}(\boldsymbol{\theta})\right] d \boldsymbol{\theta} d y+\int p(\boldsymbol{\theta}) \ln [p(\boldsymbol{\theta})] d \boldsymbol{\theta} \tag{S10}
\end{equation*}
$$

Expanding the logarithm,

$$
\begin{align*}
U(\boldsymbol{d})=- & \left.\iint p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) p(\boldsymbol{\theta}) \ln p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) d \boldsymbol{\theta} d y-\iint p(y) \mid \boldsymbol{\theta}, \boldsymbol{d}\right) p(\boldsymbol{\theta}) \ln p(\boldsymbol{\theta}) d \boldsymbol{\theta} d y  \tag{S11}\\
& +\iint p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) \mathrm{p}(\boldsymbol{\theta}) \ln p(y \mid \boldsymbol{d}) d \boldsymbol{\theta} d y+\int p(\boldsymbol{\theta}) \ln [p(\boldsymbol{\theta})] d \boldsymbol{\theta}
\end{align*}
$$

The $y$ integral in the $2^{\text {nd }}$ term amounts to 1 , so the $2^{\text {nd }}$ and $4^{\text {th }}$ terms cancel, yielding

$$
\begin{equation*}
\mathrm{U}(\boldsymbol{d})=-\int\left\{\int p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) \ln p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) d y\right\} \mathrm{p}(\boldsymbol{\theta}) d \boldsymbol{\theta}+\iint p(y \mid \boldsymbol{d}) \ln p(y \mid \boldsymbol{d}) d y \tag{S12}
\end{equation*}
$$

Recalling earlier expressions, $p(y \mid \boldsymbol{\theta}, \boldsymbol{d})$ is essentially the distribution of noise, so the first term is the entropy of the measurement noise distribution, averaged over parameters. The second term is the information entropy of $p(y \mid \boldsymbol{d})$, the $y$ distribution with only the design given. $p(y \mid \boldsymbol{d})=\int p(y \mid \boldsymbol{\theta}, \boldsymbol{d}) p(\theta) d \boldsymbol{\theta}$, or more explicitly,

$$
\begin{equation*}
p(y \mid \boldsymbol{d})=\int p_{\eta}(y-f(\boldsymbol{\theta}, \boldsymbol{d})) p(\boldsymbol{\theta}) d \boldsymbol{\theta} . \tag{S13}
\end{equation*}
$$

This expression shows that $p(y \mid \boldsymbol{d})$ is a convolution of the noise distribution and the distribution of model values due to the parameter distribution. Loosely, [S10] suggests that the highest utility will be made with designs where random draws from the parameter distribution produce the largest variations in the model function results.

## S.2.4 Evaluation of Utility Function

The double integrals in the entropy loss [S10] are potentially quite expensive. Both involve integration over both the noise distribution and the parameter distribution. However, there are several factors that relax the need for precise evaluation of the utility.

- If the utility is only used to select designs $d$ where $U(d)$ is large or maximum, precise evaluation is not needed. Only the relative magnitude of $U(d)$ is important for selecting candidate $d$ values.
- Further, precision in selecting $d$ is also non-critical in many cases. All measurements decrease information entropy in expectation, so sloppy selection of $d$ only affects the efficiency of a measurement, not the validity of the measurement results.
- In many common applications, the measurement noise does not depend on model parameters $\theta$, simplifying the first term in [S10].

In view of these factors, the optbayesexpt software adopts two approximations that dramatically reduce the computational cost of evaluating the utility.

For many common distributions, the information entropy has the form $\ln w+C$ where $w$ is a parameter describing the width of the distribution. A much smaller sample is needed to estimate the width of a distribution than to estimate the information entropy. The width of $p(y \mid d)$ described by the convolution in $[\mathrm{S} 11]$ is approximated by the standard deviation of the noise distribution $\sigma_{\eta}$, and the standard deviation $\sigma_{\theta}$ of the $f(\theta, d) p(\theta)$ distribution, summed in quadrature:

$$
\begin{equation*}
H(y \mid d) \approx \frac{1}{2} \ln \left(\sigma_{\eta}^{2}+\sigma_{\theta}^{2}\right) \tag{S14}
\end{equation*}
$$

To ensure smoothness of $U(d)$, the same draws from $p(\theta)$ are used to form and estimate $U^{*}(d)$ for all values of $d$. When $f(\theta, d)$ is a smooth function of $d$ for fixed $\theta$, the estimate $U^{*}(d)$ will also be smooth. A small sample (tens) are drawn from $p(\theta)$ to estimate the width of distributions.

## S. 3 IMPLEMENTATION OF PROBABILITY DISTRIBUTIONS

We use sequential Monte Carlo (SMC) methods to provide a computer-friendly approximation to analytical probability distributions. The distribution $p\left(\theta_{1}, \theta_{2}, \ldots, \theta_{k}\right)$ is represented by $N$ samples $\boldsymbol{\theta}_{i}=\left\{\theta_{1, i}, \theta_{2, i}, \ldots, \theta_{k, i}\right\}, i=1, \ldots, N$. Each sample can be regarded as the coordinates of a particle in $k$ dimensional parameter space and the ensemble of particles as a cloud or swarm. Each particle is also assigned a weight $w_{i}$, so that the probability density is represented by the weighted density of points in $\theta$ space. Computationally, the distribution is implemented by a dimension $N \times k$ array listing the particle coordinates and a length $N$ array listing the corresponding weights.

To incorporate new data $y_{n+1}$ using Bayesian inference, the likelihood of the result $p\left(y_{n+1} \mid \theta_{i, n}, d_{n+1}\right)$ initially modifies the weights, but does not affect the particle coordinates.

$$
\begin{gathered}
W_{i, n+1}=p\left(y_{n+1} \mid \theta_{i, n}, d_{n+1}\right) w_{i, n} \\
w_{i, n+1}=W_{i, n+1} /\left(\Sigma W_{i, n+1}\right)
\end{gathered}
$$

This Bayesian inference process will tend to decrease weights for low-probability regions in $\theta$ space to very small values, eventually leaving a small number of particles to represent higher-probability regions. To circumvent this problem, SMC methods typically use a resampling method that effectively reassigns particles into high-probability regions.

After each inference step, the effective number of particles, $N_{\text {eff }}=1 / \Sigma w_{i}^{2}$ is calculated. If $N_{\text {eff }}$ is less than (typically) half of $N$, the resampling procedure is executed as follows:

1. $N$ particles are chosen with probability $w_{i}$ from the current distribution with replacement. Some particles may be chosen more than once, some once, and those that are not chosen are abandoned.
2. To separate particles that were chosen multiple times, each of the particles is given a random displacement that is small compared to the distribution's standard deviation. Then, to compensate for the diffusion that this random displacement produces, all particles are contracted slightly toward the distribution's mean value.
3. Finally, each particle weight is assigned a uniform value $w_{i}=1 / N$.

## S. 4 SPECIFICATIONS OF THE COMPUTATIONAL HARDWARE USED FOR SEQUENTIAL BAYESIAN EXPERIMENT DESIGN

All sequential Bayesian experiment design calculations were performed on a single core (one thread) of Intel Xeon Processor E3-1225 v2 @3.20 GHz [8] using the optbayesexpt python package [9]. The wall-clock measurement times are estimated from data file modification times. The data saving was performed eight times more often for sequential Bayesian experiment design compared with the conventional setup (every 1000 points, instead of every scan of 8000 points). Moreover, the file modification times do not account for the additional time required to fit the conventional data. Hence, the $36 \%$ per-measurement slowdown associated with the Bayesian computational time is an upper bound and the difference in total throughput may be smaller.

## S. 5 SPEEDUP OF THE SEQUENTIAL BAYESIAN EXPERIMENT DESIGN

As discussed in the manuscript, a big factor that influences the speedup of the sequential Bayesian experiment design is the fraction of settings space occupied by the signal, compared with the whole scanning or sensing range. This ratio of signal to sensing range can vary significantly depending on the task at hand, and can be both much smaller than in our study (for example, in magnetometers/sensors with broad sensing range) leading to even larger speedup, or bigger, leading to a smaller speedup. In any case, the Bayesian algorithm is going to outperform the conventional scan-and-average technique. Here, we consider the "worst case" scenario for a gained speedup, when an experimenter can guess the minimal size and location of the scanning range just after one quick scan and adjust the settings appropriately. For the experiment described in the manuscript, the lower bound
for such a range would be about 16 MHz , or a tenth of the scanning range used in the manuscript ( 8 MHz is occupied by the dips, plus at least 4 MHz on each side for the shoulders). Such scanning range would allow the conventional method to collect ten times more measurements in the signal area in the same period of time, and potentially making it up to ten times faster. Even if one assumes no increase in the Bayesian algorithm speed with smaller range, the sequential Bayesian experiment design will still be 4.5 times faster than the conventional measurement. A milestone of speedup by more than a factor of 2 is very likely to be of practical relevance in any measurement, technology or business. Hence-even for applications where one, in a way, already knows the answer and can guess the location and size of the minimal scanning range-the speedup that can be achieved using sequential Bayesian experiment design surpasses this milestone by a huge margin.

## REFERENCES

[1] A. Gruber, A. Drabenstedt, C. Tietz, L. Fleury, J. Wrachtrup, and C. von Borczyskowski, Scanning confocal optical microscopy and magnetic resonance on single defect centers, Science 276, 2012 (1997).
[2] G. Davies and M. F. Hamer, Optical studies of the 1.945 eV vibronic band in diamond, Proc. R. Soc. London. A. Math. Phys. Sci. 348, 285 (1976).
[3] K. Chaloner and I. Verdinelli, Bayesian experimental design: a review, Stat. Sci. 10, 273 (1995).
[4] X. Huan and Y. M. Marzouk, Simulation-based optimal Bayesian experimental design for nonlinear systems, J. Comput. Phys. 232, 288 (2013).
[5] T. Bayes, An essay towards solving a problem in the doctrine of chances, Philos. Trans. R. Soc. London 53, 370 (1763).
[6] P. S. Laplace, Memoire sur la probabilite des causes par les evenemens, Mem. Math. Phys. 6, 621 (1774).
[7] P. S. Laplace, Memoir on the probability of the causes of events, Stat. Sci. 1, 364 (1986).
[8] Disclosure: certain commercial equipment, instruments, or materials are identified in this paper in order to specify the experimental procedure adequately. Such identification is not intended to imply recommendation or endorsement by NIST, nor is it intended to imply that the materials or equipment identified are necessarily the best available for the purpose.
[9] R.D. McMichael, Optimal Bayesian Experiment Design Software [Online] (2020). https://github.com/usnistgov/optbayesexpt/
R.D. McMichael, Optimal Bayesian Experiment Design Documentation [Online] (2020). https://pages.nist.gov/optbayesexpt/

