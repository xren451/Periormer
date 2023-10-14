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

For multiple sins functions on the $i_{th}$ signal, the mixture PDF will be:
$$f(x) = \sum_{i=1}^{N} \frac{p(x,i)}{N} $$

Theorem 2：
$$0 \leq Var(\mathcal{X}^{h}(0,Period(L)),\mathcal{X}^{h}(0,LCM(Period(1,2,...L)))) \leq \frac{L_{n}}{1-0}\int_{0}^{1} \left[A\sin(\omega t) - \frac{A}{\omega}(1 - \cos(\omega))\right]^2 dt$$
and
$$0 \leq E(\mathcal{X}^{h}(0,Period(L)),\mathcal{X}^{h}(0,LCM(Period(1,2,...L)))) \leq \frac{L_{n}}{1-0}\int_{0}^{1} A\sin(\omega t) dt$$

giving $L_{n}$ as the number of terms which cannot be divided by the largest period L.
