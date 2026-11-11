# Goodness_of_fit-on-auto-claims
A small project comparing empirical data against an approximated distribution.
# Gamma Distribution Parameter Estimation & Goodness-of-Fit on Auto Claims

A numerical analysis project in Python to model auto insurance claim amounts using a two-parameter Gamma distribution $\Gamma(\alpha, \beta)$. The pipeline estimates parameters via Maximum Likelihood Estimation (MLE) using Halley's / second-order modified Newton-Raphson method, evaluates empirical fit, and runs a Kolmogorov-Smirnov test computed through iterative composite trapezoidal numerical integration.

---

## 📌 Overview

Modeling claim severity is a core task in actuarial science and risk management. This project analyzes real collision claims (`Single Vehicle Collision` and `Multi-vehicle Collision`), fits a continuous Gamma distribution, and statistically validates the fit:

1. **Data Filtering**: Extract claim severity (`total_claim_amount`) specifically for vehicle collisions.
2. **Method of Moments Initialization**: Compute initial parameter guesses $\alpha_0$ and $\beta_0$.
3. **Modified Newton-Raphson (Halley's Method)**: Solve the nonlinear digamma equation derived from the log-likelihood derivative to find $\hat{\alpha}_{\text{MLE}}$.
4. **Numerical Integration for CDF**: Approximate the cumulative distribution function $F(x) = \int_0^x f(t) \, dt$ from scratch using an iterative composite trapezoid rule.
5. **Kolmogorov-Smirnov Test**: Evaluate goodness-of-fit against empirical distribution $F_n(x)$ without relying on pre-built CDF black boxes.

---

## 📐 Mathematical Formulation

### 1. Gamma PDF & Likelihood
The probability density function parameterized with shape $\alpha$ and scale $\beta$ ($\theta$) is:

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1} e^{-x/\beta}}{\beta^\alpha \Gamma(\alpha)}, \quad x > 0$$

Given sample mean $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$, the MLE for scale is linked to shape by:

$$\hat{\beta} = \frac{\bar{x}}{\hat{\alpha}}$$

Maximizing the profile log-likelihood leaves a single nonlinear root-finding problem for $\alpha$:

$$g(\alpha) = \ln(\alpha) - \psi(\alpha) - C = 0$$

Where:
* $\psi(\alpha) = \frac{d}{d\alpha} \ln \Gamma(\alpha)$ is the digamma function.
* $C = \ln(\bar{x}) - \frac{1}{n} \sum_{i=1}^n \ln(x_i)$.

### 2. Root-Finding (Second-Order Householder / Halley's Step)
To ensure rapid cubic convergence, the iteration step incorporates the polygamma functions $\psi'(\alpha)$ and $\psi''(\alpha)$:

$$\alpha_{k+1} = \alpha_k - \frac{g(\alpha_k) g'(\alpha_k)}{[g'(\alpha_k)]^2 - g(\alpha_k) g''(\alpha_k)}$$

* $g'(\alpha) = \frac{1}{\alpha} - \psi_1(\alpha)$
* $g''(\alpha) = -\frac{1}{\alpha^2} - \psi_2(\alpha)$

### 3. Kolmogorov-Smirnov Test via Trapezoidal Quadrature
To test $H_0: F(x) = F_0(x)$, we calculate the test statistic:

$$D = \sup_{x} |F_n(x) - F(x)|$$

Instead of calling pre-packaged CDF solvers, $F(x_{(i)})$ is evaluated piece-wise using composite trapezoidal integration across sorted claim increments:

$$\int_a^b f(t)\, dt \approx \frac{h}{2} \left[ f(a) + 2\sum_{j=1}^{m-1} f(a + jh) + f(b) \right]$$

The null hypothesis is rejected at significance level $\alpha = 0.05$ if:

$$D > \frac{1.36}{\sqrt{n}}$$

---

## 🛠️ Requirements & Setup

Ensure you have Python 3.9+ installed along with the required scientific libraries:

```bash
pip install numpy pandas scipy matplotlib
