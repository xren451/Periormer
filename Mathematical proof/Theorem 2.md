Theorem 2：
$$0 \leq E\left [\mathcal{X}^h(1,L_{max}) - \mathcal{X}^h(1, L_{lcm})\right] \leq  \sum _{i=1}^{L_{n}} \int_0^1 A_{i} \sin (\omega_{i} t) d t
$$

\begin{align*}
0 \leq &\operatorname{Var}\left[\mathcal{X}^h(1,L_{max}) - \mathcal{X}^h(1, L_{lcm})\right] \\ 
& \leq  \sum _{i=1}^{L_{n}} \int_0^1\left[A_{i} \sin (\omega_{i} t)-\frac{A_{i}}{\omega_{i}}(1-\cos (\omega_{i}))\right]^2 d t,
\end{align*}
%
where $A_{i}$ and $\omega_{i}$ are the amplitude and phase of $i$-th periodic signal and $L_{n}$ represents the number of terms that cannot divide the largest period $L_{max}$. 


Theorem 2 denotes the estimated expectation and variance following by $L_{period}$ compared to LCM. Under this theorem, we can calculate the information loss.

Proof.
To prove the lower bound. 

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

Due to the linear characteristics of the expectations , we get:

 $$E[x(t)]=E\left[x_1(t)+x_2(t)\right]$$
 
$$Var[x(t)]=Var\left[x_1(t)+x_2(t)\right]=Var\left[x_1(t)\right]+Var\left[x_2(t)\right]=\frac{A^2}{2}+\frac{B^2}{2}=\frac{A^2+B^2}{2}$$

If all of the terms $N$ can be divided by Period (L), then  $ 0 \leq E(\mathcal{X}^{h}(0,Period(L)),\mathcal{X}^{h}(0,LCM(Period(1,2,...L)))) $ as all periodic functions are shown without any information loss. Then the expectations of \mathcal{X}^{h}(0,Period(L)) and \mathcal{X}^{h}(0,LCM(Period(1,2,...L)) will be zero.

Similarly, due to the linear characteristics of variance, we get:

$$Var[x(t)]=Var\left[x_1(t)+x_2(t)\right]=Var\left[x_1(t)\right]+Var\left[x_2(t)\right]=\frac{A^2}{2}+\frac{B^2}{2}=\frac{A^2+B^2}{2}$$

Then the variance of \mathcal{X}^{h}(0,Period(L)) and \mathcal{X}^{h}(0,LCM(Period(1,2,...L)) will be the same. The lower bound is proved.

To prove the upper bound.

The worst case will be there will be L_{n} terms which can not be divided by L_{period}, and there will be at most 1 period for every terms.
Therefore, we only need to calculate the information loss in one period of each periodic term.

To achieve this, we calculate the average value of sins function $A \sin (\omega t)$ in the range $[0,1]$ as:
$$E=\frac{1}{1-0} \int_0^1 A \sin (\omega t) d t $$
 where $A$ and $\omega$ are constant, then we can get
$$E=\frac{A}{\omega}[-\cos (\omega \cdot 1)+\cos (\omega \cdot 0)]$$.
Due to $\cos (0)=1$ and $\cos (\omega)$ are determined by $\omega$, then $E$ :
$$E=\frac{A}{\omega}(1-\cos (\omega))$$

To calculate the variance of sins function $A \sin (\omega t)$ in the range $[0,1]$, we get:
$$\sigma^2=\frac{1}{1-0} \int_0^1[A \sin (\omega t)-\mu]^2 d t$$
We can also get:
$$\sigma^2=\frac{1}{1-0} \int_0^1\left[A \sin (\omega t)-\frac{A}{\omega}(1-\cos (\omega))\right]^2 d t$$

If all $L_{n}$ terms lack one period, then we can obtain information loss by multiplying $L_{n}$.
Then upper bound is proved.

