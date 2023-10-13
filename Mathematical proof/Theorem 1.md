Theorem 1: $$|\mathcal{X}-\mathcal{X}^{\mathrm{h}}| \leq M* \sqrt{R^{2}}, giving N \propto R^{2}$$.

Proof:

Following Discrete Fourier Transform (DFT) on the discrete signal,
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi}{N}kn}$$

where $N$ is the number of terms and $k$ is the frequency index. We introduce 
$R^2 = 1 - \frac{SSR}{SST}$, as SSR represents the the residual sum of squares and SST represents the total sum of squares. SST is the constant and SSR 
is determined by term $N$ in the Fourier transform, therefore, $R^{2} \propto N$.

We can get   
$$X^{\text{h}}[k] = \sum_{n=0}^{N-1} X^{\text{h}}[n] \cdot e^{-j\frac{2\pi}{N}kn}$$ and 
$$X^{l}[k] = \sum_{n=0}^{N-1} X^{l}[n] \cdot e^{-j\frac{2\pi}{N}kn}$$
According to Cauchy-Schwarz inequality， 

$$|X^{l}[k]| \leq \sqrt{\sum_{n=0}^{N-1} |X^{l}[n]|^2} \cdot \sqrt{\sum_{n=0}^{N-1} |e^{-j\frac{2\pi}{N}kn}|^2}$$

where equality holds when $X^{l}[n]$ and $e^{-j\frac{2\pi}{N}kn}$ have the same phase.
Assuming low component $X^{l}[n]$ is a truncation error term and has an upper bound represented by a constant $M$, i.e., $X^{l}[n] \leq M$ for all $n$, we have:

$$|X^{l}[k]| \leq M \cdot \sqrt{\sum_{n=0}^{N-1} |e^{-j\frac{2\pi}{N}kn}|^2}$$

Notice that $|e^{-j\frac{2\pi}{N}kn}|^2$ is the square of a sine function. This integral can be bounded by a constant less than 1, i.e., $|e^{-j\frac{2\pi}{N}kn}|^2 \leq 1$.

Therefore, as $R^{2} \propto N$, we can obtain: $$|X^{l}[k]| \leq M \cdot \sqrt{\sum_{n=0}^{N-1} 1}$$

$$|X^{l}[k]| \leq M \cdot \sqrt{N}$$,  $$|X^{l}[k]| \leq M \cdot R^{2}$$.
