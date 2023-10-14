The PDF of Sins signal:

Due to sins signal is a periodic signal, we should firstly calculate the probability density function(PDF). Assume the period of the sins function is $T$.
As we know:

$$x(t+T)=A\sin(\omega(t+T)+\phi)=A\sin(\omega t+2\pi k +\omega T+\phi)=A\sin(\omega t+\phi)$$
Therefore, any two positions in the same period share the same distribution of probability.

We can get the expectations as follows:

$$\begin{aligned}E[x(t)]&=\frac{1}{T}\int_{t_{o}}^{t_0+T}A\sin(\omega t)\mathrm{d}t=0\\
E[{x(t)}^{2}]&=\frac{1}{T}\int_{t_{o}}^{t_0+T}A^{2}\sin^{2}(\omega t)\mathrm{d}t=\frac{A^{2}}{2}\\
\end{aligned} $$

Therefore, the PDF in the period will be:
$$p(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{x^{2}}{2\sigma^{2}}\right)$$, where $\sigma^{2}=\frac{A^{2}}{2}$.

For the whole time points $(-\infty,\infty)$, the periodic signals can appear in any range. Therefore, the PDF in the whole time range will be:
$$p(x)=\frac{1}{\sqrt{2\pi}\sigma}\sum_{k=-\infty}^{\infty}\exp\left(-\frac{(x-A\sin(\omega t_k))^{2}}{2\sigma ^{2}}\right)$$

For multiple sins functions, the mixture PDF will be:
$$f(x) = \sum_{i=1}^n p_i \cdot \frac{1}{2\pi\sigma_i} \cdot \exp\left(-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right) \cdot \sin(2\pi f_i x + \phi_i)$$
