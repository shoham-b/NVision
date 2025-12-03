2

## 1




## Engineering Project Course
Enhancing NV Center Magnetometry
with Bayesian Method



Submission date: November 2024
Submitted by:
Shir Giat        315523639       shir.giat@mail.huji.ac.il


## Supervisor:
Prof. Nir Bar-Gill, Department of Applied Physics, The Hebrew
University of Jerusalem

## 2

Table of Contents


- Abstract ------------------------------------------------------------------------------------------  4

- Introduction ------------------------------------------------------------------------------------  5-6
2.1 Background ---------------------------------------------------------------------------------------  5
2.2 Objectives -----------------------------------------------------------------------------------------  5
2.3 Scope -----------------------------------------------------------------------------------------------  5
2.4 Organization --------------------------------------------------------------------------------------  6

-   Literature and state-of-the-art review --------------------------------------------------  6-7
3.1 Summary of Existing Work --------------------------------------------------------------------  6
3.2 Gaps in the Literature ------------------------------------------------------------------------  6-7
3.3 Addressing the Gaps ---------------------------------------------------------------------------  7

-   Theoretical Background ------------------------------------------------------------------  7-12
4.1. Nitrogen-Vacancy (NV) Center -----------------------------------------------------------  7-8
4.2 Zeeman splitting ------------------------------------------------------------------------------  8-9
4.3 Optically Detected Magnetic Resonance (ODMR) --------------------------------------  9
4.4 Lorentzian Resonance Model -----------------------------------------------------------  9-10
4.5 Electron Spin Resonance (ESR) Model ---------------------------------------------------  11
4.6 Bayesian Method -----------------------------------------------------------------------------  12

-  Methodology and implementation -------------------------------------------------- 12-21
5.1 Design and Development ------------------------------------------------------------------  12
5.2 Tools and Equipment ------------------------------------------------------------------------  12
5.3 Procedures and Algorithms ------------------------------------------------------------ 13-18

## 3

5.4 Block diagrams ------------------------------------------------------------------------------  19-20
5.5 Software Implementation ----------------------------------------------------------------  20-21
5.6 Challenges and Solutions ---------------------------------------------------------------------  21

-   Testing and Results ------------------------------------------------------------------------- 21-27
6.1 Testing Procedures -------------------------------------------------------------------------  21-22
6.2 Results ----------------------------------------------------------------------------------------  22-26
6.3 Analysis -------------------------------------------------------------------------------------------  26
6.4 Performance Evaluation ------------------------------------------------------------------  26-27

-  Discussion ------------------------------------------------------------------------------------  27-28

- Conclusions ---------------------------------------------------------------------------------------  28

- References ----------------------------------------------------------------------------------------  29





## 4

## 1 Abstract

This project aimed to enhance magnetic field measurements using nitrogen-vacancy
(NV) centers in diamonds. The primary goal was to improve the precision, accuracy,
and efficiency of Optically Detected Magnetic Resonance (ODMR) measurements by
implementing a sequential Bayesian method.

## Methodology:
- Developed and implemented a Bayesian algorithm for ODMR signal analysis.
- Conducted extensive simulations using Electron Spin Resonance (ESR) models to
compare the Bayesian approach with the conventional ODMR technique.
- Performed comprehensive analysis covering: ESR Model Comparison, Signal-to-
Noise Ratio (SNR) vs. Noise level analysis, Root Mean Squared Error (RMSE) vs.
Noise level analysis, and Convergence Rate Assessment.

Results demonstrated significant advantages of the Bayesian method:
- Improved precision in estimating the central frequency of the ESR model, achieving
concentrated samples with fewer measurements.
- Enhanced sensitivity to small variations in central frequency and peak width,
enabling detection of subtle magnetic field changes.
- Superior noise robustness, maintaining a higher SNR  and exhibiting a more gradual
increase in RMSE as noise levels increased.
- Faster convergence to accurate model parameters, requiring significantly fewer
measurements even at higher noise levels.
These findings highlight the Bayesian method's potential to substantially enhance
magnetic field characterization using NV centers in diamonds, offering improved
precision, accuracy, and efficiency compared to the conventional ODMR technique.
This advancement could have significant implications for various applications in
quantum sensing and magnetometry.




## 5

## 2   Introduction

## 2.1   Background
Nitrogen-Vacancy (NV) centers in diamonds have emerged as a powerful tool in
quantum  technology,  particularly  for  magnetometry.  These  atomic-scale  defects
possess unique spin properties, including spin-state-dependent fluorescence and long
coherence times, making them ideal for high-precision magnetic field measurements.
The current standard technique, Optically Detected Magnetic Resonance (ODMR),
while effective, is affected by noise measurement issues and is not very efficient  ,
leading to less accurate results. These limitations have urged the exploration of
advanced techniques to enhance measurement accuracy and efficiency in NV center
magnetometry.

## 2.2   Objectives
This project aims to:
- Implement and evaluate the Bayesian method for NV center magnetometry.
- Compare the performance of the Bayesian method against the conventional ODMR
technique in terms of precision, efficiency, and noise reduction.
- Assess the Bayesian method's effectiveness in determining related parameters
under various conditions.

## 2.3   Scope
The project encompasses the following:
- Theoretical modeling of NV center behavior using Lorentzian Resonance and
Electron Spin Resonance (ESR) models.
- Implementation  of the ODMR  and  Bayesian  method  for  magnetic  field
measurements.
- Simulation-based comparison of the two methods.
- Analysis of method performance focusing on robustness to different noise levels,
sensitivity to parameter variations, and convergence speed.
- Evaluation of the methods' ability to accurately determine the center resonance
frequency and width parameter.


## 6

## 2.4   Organization
The project is structured as follows:
- Introduction: Provides context, objectives, scope, and organization.
- Theoretical Background: Covers the physics of NV centers, including their structure,
energy levels, and spin-dependent fluorescence.
-  Measurement  Techniques:  Details  the  ODMR  technique  and  introduces  the
sequential Bayesian method.
- Methodology: Describes the implementation of both methods, including simulation
parameters and data analysis techniques.
- Results and Discussion: Presents a comparative analysis of the ODMR and Bayesian
methods, focusing on their performance.
- Conclusion: Summarizes key findings and discusses implications for future NV center
magnetometry applications.


3 Literature and state-of-the-art review

3.1   Summary of Existing Work
Recent advancements in NV-based magnetometry have shown significant progress.
Schloss et al. (2018) demonstrated simultaneous measurement of multiple NV centers,
enhancing overall magnetometry sensitivity
## 1
. Dushenko et al. (2020) introduced a
sequential Bayesian experiment design that dramatically reduced measurement times
compared to traditional frequency-swept methods
## 2
## .

3.2   Gaps in the Literature
Despite these advancements, several key gaps remain in the field:
- Limited noise analysis: Current methods lack thorough performance evaluation
under varying noise conditions, particularly in environmentally noisy settings.

## 1
https://doi.org/10.1103/PhysRevApplied.10.034044
## 2
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.054036

## 7

- Trade-off between sensitivity and resolution: While multi-NV center
measurements enhance sensitivity, they often reduce spatial resolution. There's
limited work on optimizing this balance.
- Narrow parameter focus: Most research concentrates on specific parameters like
frequency estimation, overlooking crucial factors such as linewidth and long-term
stability.

3.3 Addressing the Gaps
Our project addresses these gaps through:
- Comprehensive noise analysis: Extensive simulations demonstrate the Bayesian
method's robustness to noise, evaluating SNR and RMSE across increasing noise
levels.
- Sensitivity-resolution optimization: We examine sensitivity to small variations in
central frequency and peak width, analyzing linewidth dependency and its impact
on parameter sensitivity.
- Rigorous Bayesian method evaluation: Thorough testing under various conditions,
including high-noise environments typical in single NV center measurements,
assessing convergence to accurate model parameters.


## 4   Theoretical Background

4.1   Nitrogen-Vacancy (NV) Center
NV centers are atomic-scale defects in diamonds, formed when two adjacent carbon
atoms in a diamond lattice are replaced by vacancy and a negatively charged nitrogen
atom. These NV centers have emerged as powerful tools in quantum technology,
including magnetometry, quantum information, sensing, computing, and imaging.
The NV center has a ground state, triplet state, singlet state, and an excited state. In
experiments, a green laser excites and polarizes NV centers; a microwave (MW)
antenna manipulates quantum states; and a detector collects emitted photons to
determine the system state. The fluorescence is spin-state dependent: excitation from
## 푚
## 푠
=0  leads to red light emission, while 푚
## 푠
=±1 can lead to either red light
emission or transition via dark states, reducing fluorescence. The external magnetic
field causes Zeeman splitting of 푚
## 푠
=±1  levels.

## 8


Figure1: A structure of a NV center in a diamond lattice. The diamond unit cell consists of a substitutional nitrogen
atom (red), an adjacent vacancy (gray), and carbon atoms (blue).


Figure2: Schematic of the energy level structure of the negatively charged NV center. Both the ground state and
the excited state are spin triplets and microwave excitation leads to non-radiative relaxation of 푚
## 푠
## =±1.

NV systems are used to measure and characterize quantum systems and as sensors to
measure external magnetic fields. Using NV centers as an atomic-size magnetic field
sensor has become a popular tool that improves measurements because of their
unique spin properties, such as their photoluminescence dependence on spin states,
long coherence time, high-resolution, and ability to operate in a wide range of
conditions.

4.2   Zeeman splitting
Zeeman splitting is a phenomenon where the energy levels of atoms or molecules split
into sub-levels when exposed to an external magnetic field. This occurs due to the
interaction between the magnetic field and the magnetic moments of electrons, which
causes the energy states to shift based on the orientation of their spins relative to the
field.

## 9

In NV centers in diamonds, Zeeman splitting is critical for magnetic field measurement.
When a magnetic field is applied, the NV center's electron spin states split into distinct
energy levels. This results in three resonance frequencies, corresponding to different
spin states, that are directly proportional to the strength and direction of the magnetic
field. By analyzing these frequencies, one can accurately determine the characteristics
of the surrounding magnetic field.


Figure3: (a) Energy level diagram of an NV center showing Zeeman splitting under the influence of an external
magnetic field. The 푚
## 푠
=±1 levels split, with energy proportional to 훾퐵
## 111
.  (b) Fluorescence intensity vs.
microwave frequency, shows two dips, indicating Zeeman splitting between the 푚
## 푠
=±1 states, corresponding to
the magnetic field strength.


4.3   Optically Detected Magnetic Resonance (ODMR)
ODMR is a technique to measure magnetic fields using NV centers in diamonds. It
works by illuminating NV centers with light while simultaneously applying microwaves,
and then monitoring changes in fluorescence. Resonance occurs at specific microwave
frequencies, which are indicated by dips in fluorescence, revealing the optically
detected magnetic resonance and providing information about the magnetic field.


## 4.4   Lorentzian Resonance Model
The Lorentzian model describes the behavior of NV centers in diamonds under a
magnetic field. The interaction between the NV center's electrons and the nitrogen
nucleus causes a splitting into three distinct photoluminescence dips. These dips,
which correspond to different nuclear spin states, are used to measure and evaluate
the effectiveness of magnetic field measurement methods.
The model is fitted to experimental data to determine key parameters, providing
insights into the magnetic field characteristics.

## 10

The equation for the normalized photon count signal μ is:
Eq. (1)     μ=1−
## 푎∙푘
## 푁푃
## (
## 푓−푓
## 0
## −∆푓
## )
## 2
## +훾
## 2
## −
## 푎
## (
## 푓−푓
## 0
## )
## 2
## +훾
## 2
## −
## 푎/푘
## 푁푃
## (
## 푓−푓
## 0
## +∆푓
## )
## 2
## +훾
## 2
## )


This equation accounts for the three resonance dips caused by the energy transitions
in the NV center under an applied magnetic field
## 3
## .

The parameters are the following:
## 푓
## 0
: The center resonance frequency, indicating the central point around which the
energy levels split.
Δ푓 : The hyperfine splitting frequency, representing the shift in resonance caused by
the interaction between the NV center's electron spin and the nitrogen nucleus's spin.
푎 : The contrast factor, which determines the depth of the resonance dips and how
much the signal deviates from the baseline.
훾 : The linewidth of the resonance, describing how broad each resonance peak is.
## 푘
## 푁푃
: The nuclear polarization factor, which represents the contribution of nuclear
polarization to the resonance pattern.








Figure4: Photoluminescence intensity (μ) vs. frequency for NV Center Magnetometry, modeled using the
## Lorentzian Resonance Model



## 3
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.054036

## 11

4.5   Electron Spin Resonance (ESR) Model
The Electron Spin Resonance (ESR) model is used to simulate and analyze ESR signals.
By fitting this Lorentzian model to experimental data, one can extract the resonance
frequency  and  linewidth,  both  are  essential  for  determining  magnetic  field
characteristics and spin interactions in the system. This is a simplified model, as it
focuses on a single resonance peak and ignores more complex interactions. However,
it is sufficient for many applications where the main goal is to estimate the magnetic
field strength and obtain a clear understanding of the central resonance without
needing to account for additional hyperfine interactions or multi-peak phenomena.
The equation for the ESR model is the following:

## Eq. (2)     퐿표푟푒푛푡푧푖푎푛
## (
## 푓;푓
## 0
## ,훾
## )
## =
## 1
## 휋
## ∙
## 훾
## (
## 푓−푓
## 0
## )
## 2
## +훾
## 2
## ∝
## 훾
## (
## 푓−푓
## 0
## )
## 2
## +훾
## 2


The parameters are the following:
## 푓
## 0
: The center resonance frequency, indicating the central point around which the
energy levels split.
훾 : The linewidth of the resonance, which determines how broad the resonance peak
is.

Figure5: ESR Signal vs. Frequency for NV Center Magnetometry


## 12

## 4.6   Bayesian Method
Sequential Bayesian experiment design is an advanced approach used to improve
precision in NV center magnetometry. It uses Bayesian inference to iteratively update
information about model parameters as new measurement data is collected. Based on
this updated information, the method selects experimental settings that maximize the
chances of refining the parameters.
This method reduces noise by relying on the Law of Large Numbers (LLN), where
repeated measurements average out random Gaussian noise to zero. The algorithm
models a distribution of possible measurement outcomes and predicts the average
improvement based on the collected data. It optimizes the experimental setup
through a utility function that selects the most informative settings.
This iterative process is repeated until a desired level of precision is reached. By
continuously adapting the experimental setup based on the updated parameter
estimates, the sequential Bayesian method ensures that resources are used efficiently,
focusing measurements on settings that are most likely to improve the accuracy of the
model. This is particularly useful for NV center magnetometry, where noise reduction
and precision are critical for effective magnetic field sensing.


5   Methodology and implementation

5.1   Design and Development
The project was developed in several stages :
- Theoretical modeling of NV center behavior.
- Implementation of the Bayesian optimization algorithm.
- Creation of a simulation environment.
- Comparative analysis of Bayesian and conventional methods.

5.2   Tools and Equipment
- Programming Language: Python (for simulation and data analysis).
- Libraries: NumPy (for numerical computations), Matplotlib (for data visualization).



## 13

5.3   Procedures and Algorithms

5.3.1    Algorithm: The Usual Method in ESR Experiments
## 1. Initialization:
- Define a range of frequencies and widths.
- Set the number of samples.
- Set the noise level value.
- Define true parameters (f
## 푡푟푢푒
## , 훾
## 푡푟푢푒
## )
- Define the new range of frequencies and widths- which is within a range around
the true parameters.
- Randomly select initial guesses (f
## 0
## ,훾
## 0
## )

- Frequency Selection: Select frequencies randomly from the new range, based on
the number of samples.

## 3. Measurement Simulation
- Use the ESR model to generate a signal based on the given frequency, peak
position, and width.
- According to the specified noise level, add Gaussian noise to the signal to simulate
real-world measurement noise.
- The resulting simulated measurement is the sum of the clean signal from the ESR
model and the added noise.

- Repetition: repeat the process for the specified number of measurements.

- Parameter estimation
Fit the points to Lorentzian function and then get estimations to the central
frequency f
est
and the linewidth γ
est



5.3.2    Algorithm: The Bayesian Method in ESR Experiments
## 1. Initialization
- Define a range of frequencies and widths.
- Set the number of samples.
- Set the noise level.
- Define true parameters (f
## 푡푟푢푒
## , 훾
## 푡푟푢푒
## )

## 14

- Define the new range of frequencies and widths- which is within a range around
the true parameters.
- Randomly select initial guesses (f
## 0
## ,훾
## 0
## )
- Randomly selects the first frequency from the new range, based on the number
of samples.

## 2. Frequency Selection

In this method, we define 푈(푓) as the utility function, 휃 as the random variable
that  is  distributed  according  to  the  probability  density  function 푝(휃) and
## 푝(휃|푦
## 푛
## ,푓
## 푛
) as  the  parameter  distribution.  In  our  context, 휃 represents  the
parameters 휃=(푓
## 0
,훾) .  The  accumulated  measurement  results  are 푦
## 푛
## =
## {푦
## 1
## ,푦
## 2
## ,...,푦
## 푛
} and  the  frequencies  are 푓
## 푛
## ={푓
## 1
## ,푓
## 2
## ,...,푓
## 푛
}. The  conditional
distribution represents the parameters after n iterations. In every (n+1) iteration,
the method uses the parameter distribution to choose the best frequency.

Use the Bayesian method to select subsequent frequencies by the following
steps:

- Calculates utility (benefit ratio) for each setting using the formula:

## 푈푡푖푙푖푡푦(푓)=∑
## (
## 푉푎푟
## 푃푎푟푎푚푡푒푟푠
## 푉푎푟
## 푁표푖푠푒
## )
## /퐶표푠푡

The utility for a given frequency 푓 determines how useful that frequency will be in
refining parameter estimates. While 푓 does not appear explicitly in the utility formula,
## 푉푎푟
## 푃푎푟푎푚푡푒푟푠
depends on 푓 through the Lorentzian model. At different frequencies,
changes in parameters will affect the predicted results differently - having a minimal
effect far from the resonance frequency (f₀) but a significant impact near it. Thus,
Utility(f) effectively varies with frequency.

The parameters of the utility formula are the following:
## 푉푎푟
## 푃푎푟푎푚푡푒푟푠
: Variance due to parameters, measures how much the model's predicted
outcomes vary for different parameter values .
## 푉푎푟
## 푁표푖푠푒
: Variance  due  to noise accounts  for  the  uncertainty  introduced  by
measurement noise .


## 15

Cost: Represents the experimental cost associated with measuring at that frequency.
Defined as default with value of 1.

## Where:

## 푉푎푟
## 푃푎푟푎푚푡푒푟푠
## =
## 1
## 푁
## ∑(
## 푦
## 푖
## −푦̅
## )
## 2
## 푁
## 푖=0
## ;   푦̅=
## 1
## 푁
## ∑
## 푦
## 푖
## 푁
## 푖=0



Such that:
푁 : The number of settings, which in this context refers to the number of different
frequencies at which measurements are taken.
## 푦
## 푖
: The measured signal at the i-th setting.
푦̅ : The average of the measured signals across all N settings.

- Create a probability distribution using the formula:
After calculating the utility for each setting, a probability distribution is created:

## 푃
## (
## 푓
## )
## =푈푡푖푙푖푡푦
## (
## 푓
## )
## 푃푖푐푘푖푛푒푠푠


Where: Pickiness determines how much the utility influences the frequency selection
process. When the pickiness value is high, the algorithm favors frequencies with the
highest utility, focusing on settings that are expected to provide the most information.

- Randomly  selects  a  setting  from  this  distribution,  where higher  probability
frequencies are more likely to be chosen.


## 3. Measurement Simulation
- Use the ESR model to generate a signal based on the given frequency, peak
position, and width.
- According to the specified noise level, add Gaussian noise to the signal to simulate
real-world measurement noise.
## 푉푎푟
## 푁표푖푠푒
## =
## (
## 푛표푖푠푒_푠푡푑
## )
## 2


## 16

- The resulting simulated measurement is the sum of the clean signal from the ESR
model and the added noise.


- Update: Update the Bayesian model with new data by the following steps:


- Outcome Evaluation: Evaluate outcomes for all parameter combinations given the
new setting.

- Likelihood Calculation: The likelihood of each parameter combination given the
new data is calculated using a Gaussian likelihood formula :

## 푃
## (
## 퐷
## |
## 휃
## )
## =
## 1
## √2휋휎
## 2
exp(−
## (
## 퐷−휇
## )
## 2
## 2휎
## 2
## )


## Where:
퐷 : The data collected from the experiment, which is the measured values after
performing the experiment at a certain setting; frequency.

휃 : The parameters being estimated; peak frequency and width in the model. These
are unknowns we are trying to refine.

휎 : The standard deviation representing the measurement noise/uncertainty. It
reflects the expected variation in the measurements due to noise.

휇 : The mean predicted value of the signal based on the current parameter
estimates θ. It’s the model’s expected outcome for each set of parameters θ at the
chosen setting.

## • Probability Distribution Update:
Using the Bayes’ Theorem:
## 푃
## (
## 휃
## |
## 퐷
## )
## =
## 푃
## (
## 퐷
## |
## 휃
## )
## 푃
## (
## 휃
## )
## 푃
## (
## 퐷
## )

## Where:
## 푃
## (
## 휃
## |
## 퐷
## )
: Posterior probability distribution, representing the updated belief about the
values of θ after observing the experimental data.

## 17


## 푃
## (
## 퐷
## |
## 휃
## )
: Likelihood of observing the data D given specific values of the parameters θ.

## 푃
## (
## 휃
## )
: Prior probability, which reflects the initial belief about the parameters θ before
any new data is collected (based on past knowledge or assumptions).

The formula is the following:

## 푃
## (
## 휃
## )
## =
## 1
## √2휋휎
## 2
exp(−
## (
## 휃−휇
## )
## 2
## 2휎
## 2
## )

This formula is a Gaussian distribution, assigns higher probabilities to values near μ,
and lower probabilities to values farther away, based on their distance from μ.


## 푃
## (
## 퐷
## )
: Normalizing constant, which ensures the posterior probabilities sum to 1. It
accounts for the total probability of observing the data across all possible parameter
values.

The formula is the following:

## 푃
## (
## 퐷
## )
## =∫푃
## (
## 퐷
## |
## 휃
## )
## 푃
## (
## 휃
## )
## 푑휃

This formula integrates over all possible parameter values to compute the total
probability of observing the data, making the posterior distribution a valid probability
distribution.

- Repetition: repeat the process for the specified number of measurements.

- Parameter estimation: Fit the points to Lorentzian function and then get
estimations to the central frequency f
est
and the linewidth γ
est



5.3.3    Calculating the SNR vs. Noise Level
Compute SNR by comparing the true signal power to the average noise power at each
noise level.

## 18

## 푆푁푅=10log
## 10
## (
## 푆푖푔푛푎푙 푃표푤푒푟
## 퐴푣푒푟푎푔푒 푁표푖푠푒 푃표푤푒푟
## )

## Where:
## 푆푖푔푛푎푙 푃표푤푒푟=
## 1
## 푁
## ∑
## (
## 푡푟푢푒_푠푖푔푛푎푙[푖]
## )
## 2
## 푁
## 푖=0

Here, N is the number of samples and true_signal[i] is the value of the signal at the ith
sample.
## 푁표푖푠푒 푃표푤푒푟=
## 1
## 푁
## ∑
## (
## 푡푟푢푒_푠푖푔푛푎푙
## [
## 푖
## ]
## −푓푖푡푡푒푑_푠푖푔푛푎푙[푖]
## )
## 2
## 푁
## 푖=0

Here, fitted_signal[i] is the value of the fitted signal at the ith sample, obtained by
applying the optimized parameters from the fit to the model.

5.3.4    Calculating the RMSE vs. Noise Level

Calculate RMSE using the following formula:
## 푅푀푆퐸=
## √
## (
## 푓
## 푒푠푡
## −푓
## 푡푟푢푒
## )
## 2
## +
## (
## 훾
## 푒푠푡
## −훾
## 푡푟푢푒
## )
## 2



5.3.5    Calculating the Convergence to the model vs. Noise Level

To check convergence to the model means to check if the difference between the true
parameter frequency and the estimated frequencies values is less than a tolerance of
0.01 for 5 consecutive iterations.
Mathematically, convergence occurs iff:
## |푓
## 푡푟푢푒
## −푓
## 푒푠푡
## (
## 푖
## )
## |≤0.01
## 푓표푟 푖=푛, 푛+1,푛+2,푛+3,푛+4




## 19

5.4   Block diagrams

5.4.1   High-level comparison



The diagrams above illustrate the fundamental difference between the usual and
Bayesian methods in ESR experiments. The usual method follows a linear approach
where frequencies are randomly selected, measurements are taken sequentially. In
contrast, the Bayesian method implements an adaptive approach with a feedback
loop. After each measurement, the parameters are updated, and this information is
used to calculate the utility of potential next measurement points. This feedback
mechanism  allows  the  Bayesian  method  to  continuously  learn  from  previous
measurements and make informed decisions about subsequent frequencies, leading
to more efficient data collection and precise parameter estimation. The feedback
arrow represents this crucial adaptive element, showing how each measurement's
results influence the selection of the next measurement point.


5.4.2   Technical details

Diagram1: ESR Experiment for the usual method

## 20




Diagram2: ESR Experiment for the Bayesian method



Diagram3: frequency selection steps



Diagram4: Update parameters steps


## 5.5    Software Implementation
The software for this project was developed using Python, focusing on efficient
numerical computations and data visualization.
Key components include:
## 1. Core Function:
`normalized_lorentzian`: Implements the normalized Lorentzian function for the ESR
model.
## 2. Bayesian Experiment Function:
`esr_bayesrun`: Handles the Bayesian experiment process, selecting measurement
points and updating the probability distribution.
- Simulation and Analysis:

## 21

The  main  script  sets up parameters,  runs  simulations  for  different numbers  of
measurements, and visualizes results.
Key coding techniques include:
- Vectorization: Numpy arrays are used for efficient computation (e.g., `np.linspace`,
## `np.zeros`).
- Scientific computing libraries: Scipy for curve fitting (`curve_fit`).
- Bayesian optimization: Utilizes the `OptBayesExpt` class for Bayesian experiment
design.
- Data visualization: Matplotlib for creating subplots and scatter plots.
- Error handling: Try-except block for handling curve fitting errors.

5.6    Challenges and Solutions
Implementing the Bayesian method was challenging due to its complexity and the
need  for  precise  parameter  tuning.  Key  difficulties  included  understanding  the
probabilistic  framework,  integrating  the  OptBayesExpt  library,  and  optimizing
hyperparameters like 'pickiness'. Despite these obstacles, a persistent effort led to a
successful implementation.


6   Testing and Results

## 6.1    Testing Procedures

6.1.2 ESR Model Comparison
- Simulated ESR spectra using both methods.
- Analyzed the concentration of sampled points around the central frequency.
- Varied sample sizes to assess method stability.

6.1.3    SNR vs. Noise Level Analysis
- Introduced varying levels of Gaussian noise to the ESR model.

## 22

- Calculated the SNR for both methods at each noise level.

6.1.4    Root Mean Squared Error (RMSE) vs. Noise Level
- Computed RMSE between estimated and true model parameters.
- Analyzed RMSE trends as noise levels increased.

## 6.1.5 Convergence Analysis
- Tracked the number of measurements required to achieve stable parameter
estimates.
- Compared convergence rates at different noise levels.


## 6.2   Results

## 6.2.2   Lorentzian Resonance Model Comparison







Figure6: Comparison between the Lorentzian resonance model for different numbers of samples, points samples
with Gaussian noise of 0.05dB
This is a simulation where measurement points were sampled and Gaussian noise was
added to each point. The noise is characterized by a standard deviation of 0.05dB. For
the complete sampling methodology, please refer to section 5.3 (Procedures and
## Algorithms).

## 23

The Bayesian method outperformed the conventional approach:
- Rapid concentration of data points around the resonance dips.
- Consistent pattern across different sample sizes, indicating stable estimations.
In contrast, the conventional method showed broader dispersion, leading to less
precise resonance dips determination.


6.2.3    ESR Model Comparison


Figure7: Comparison between the ESR model for different numbers of samples, points samples with Gaussian
noise of 0.05dB

This is a simulation where measurement points were sampled and Gaussian noise was
added to each point. The noise is characterized by a standard deviation of 0.05dB. For
the complete sampling methodology, please refer to section 5.3 (Procedures and
## Algorithms).



## 24

The Bayesian method outperformed the conventional approach:
- Rapid concentration of data points around the central frequency.
- Consistent pattern across different sample sizes, indicating stable estimations.
In contrast, the conventional method showed broader dispersion, leading to less
precise central frequency determination.


6.2.4    SNR vs. Noise Level Analysis


Figure8: SNR vs. noise level for the ESR model

The Bayesian method exhibited robust performance against noise:
- Maintained higher SNR as noise levels increased.
- Showed a smoother, more predictable SNR decline with increasing noise.
In contrast, the conventional method displayed less stable SNR trends and lower
overall values.





## 25

6.2.5    RMSE vs. Noise Level Analysis

Figure9: RMSE vs. noise level of the model

The Bayesian method demonstrated superior noise resilience:
- Maintained lower RMSE despite increasing noise levels.
- Exhibited a less steep rate of RMSE increase.
- Indicated enhanced robustness in noisy environments.
In contrast, the conventional method showed higher RMSE values, steeper increases
with noise, and greater sensitivity to noise.


## 6.2.6    Convergence Analysis








Figure10: Convergence of the model

## 26

The Bayesian method showed faster and more stable convergence:
- Achieved accurate ESR signal modeling with significantly fewer measurements.
- Demonstrated lower sensitivity to noise levels during convergence.
In contrast, the conventional method required more measurements and showed
higher noise sensitivity.


## 6.3   Analysis
- Precision and Accuracy: The Bayesian method consistently provided more precise
and accurate estimates of the ESR model, crucial for accurate magnetic field
characterization.

- Noise  Resilience:  Superior  performance  in  noisy  environments  indicates  the
Bayesian method's potential for real-world applications where signal noise is a
significant challenge.

- Efficiency: Faster convergence with fewer measurements suggests the Bayesian
method could significantly reduce experimental time.

- Stability: Consistent performance across different sample sizes and noise levels
demonstrates the robustness of the Bayesian approach.



## 6.4  Performance Evaluation

## 6.4.1    Analysis

The results demonstrate significant improvements achieved by the Bayesian method
in NV center magnetometry :

- Precision Enhancement : The Bayesian method's 3x improvement in SNR (15 dB vs.
5 dB) indicates substantially clearer signal detection. This enhanced precision
allows for more accurate magnetic field measurements, potentially enabling the
detection of weaker or more subtle field variations.

## 27


- Accuracy Improvement : The 4x reduction in RMSE (8 dB vs. 2 dB) surpassed
expectations, showcasing the Bayesian method's superior parameter estimation
capabilities. This dramatic improvement in accuracy could significantly enhance
the reliability of magnetic field characterization in various applications.

- Efficiency Gain: Achieving convergence in half the measurements (60 vs. 120)
demonstrates the Bayesian method's ability to extract meaningful information
more efficiently. This could lead to faster experimental processes and reduced
resource requirements in practical applications .

- Robust Noise Handling : The method's superior performance even at high noise
levels is particularly significant. It suggests that the Bayesian approach could
extend  the  usability  of  NV  center  magnetometry  to  more  challenging
environments or enable measurements in previously inaccessible conditions.


## 6.4.2    Significance

The comprehensive improvements across precision, accuracy, efficiency, and noise
resilience indicate a potential paradigm shift in NV center magnetometry techniques.
These  advancements  could  expand  the  application  scope  of  quantum  sensing,
enabling more sensitive and reliable measurements in fields such as materials science
and quantum computing.

In conclusion, the Bayesian method's performance not only met but often exceeded
the initial objectives, demonstrating its potential to significantly advance the field of
quantum sensing and NV center magnetometry.


## 7 Discussion
The results demonstrate the significant potential of the Bayesian method in enhancing
NV-center magnetometry. The method's ability to concentrate measurements around
critical frequencies and maintain high performance in noisy conditions addresses key
challenges in quantum sensing applications.
One unexpected outcome was the degree of improvement in noisy environments.
While we anticipated better performance, the Bayesian method's ability to maintain
high SNR and low RMSE even at extreme noise levels exceeded our expectations. This
suggests  potential  applications  in  challenging  sensing  environments  previously
considered impractical for NV-center magnetometry.

## 28

Future work could focus on:
## 1. Expanding Applications:
- Broader testing across various scenarios.
- Integration with other quantum sensing technologies.
## 2. Utilizing Functional Dependencies:
- Exploring different parameter relationships.
- Adapting the method for multi-parameter estimation.
- Investigating cross-domain applications in quantum sensing.


## 8 Conclusions
This project has demonstrated the substantial benefits of applying Bayesian methods
to NV-center magnetometry. Key findings include a 3x improvement in SNR, a 4x
reduction in RMSE, and a 2x increase in measurement convergence compared to
conventional  methods. These improvements  significantly  enhance  the  precision,
accuracy, and speed of magnetic field measurements using NV centers.
The project's primary contribution is the development of a robust, noise-resistant
approach to quantum sensing, potentially expanding the application range of NV-
center magnetometry. This work lays the foundation for more efficient and accurate
quantum sensors, with implications for fields ranging from materials science to
quantum technologies.
Reflecting on the project, we've gained valuable insights into the power of adaptive
measurement strategies in quantum systems. The success of the Bayesian approach
underscores the importance of intelligent data acquisition and analysis techniques in
pushing the boundaries of quantum sensing technology.







## 29

## 10   References

## Section 3.1
https://doi.org/10.1103/PhysRevApplied.10.034044
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.054036

## Section 4.4
https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.054036
