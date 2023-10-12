Theorem 1: $$|\mathcal{X}-\mathcal{X}^{\mathrm{high}}| \leq M* R^{2}, giving N \propto R^{2}$$.

Proof:

Following Discrete Fourier Transform (DFT) on the discrete signal,
$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi}{N}kn}$$

where $N$ is the number of terms and $k$ is the frequency index. We introduce 
$$R^2 = 1 - \frac{SSR}{SST}$$, as SSR represents the the residual sum of squares and SST represents the total sum of squares. SST is the constant and SSR $$\mathcal{X}^{high}$$ 
is determined by term $$N$$ in the Fourier transform, therefore, $$R^{2} \propto N$$.
3. Combine equation ~\ref{eq:high and low decomp} and DFT, we can get   
$$
X_{\text{high}}[k] = \sum_{n=0}^{N-1} X^{\text{high}}[n] \cdot e^{-j\frac{2\pi}{N}kn}
$$ and 
$$
X^{low}[k] = \sum_{n=0}^{N-1} X^{low}[n] \cdot e^{-j\frac{2\pi}{N}kn}
$$
4. According to Cauchy-Schwarz inequality， 

$$
|X^{low}[k]| \leq \sqrt{\sum_{n=0}^{N-1} |X^{low}[n]|^2} \cdot \sqrt{\sum_{n=0}^{N-1} |e^{-j\frac{2\pi}{N}kn}|^2}
$$

where equality holds when $$X^{low}[n]$$ and $$e^{-j\frac{2\pi}{N}kn}$$ have the same phase.
Assuming low component $$X^{low}[n]$$ is a truncation error term and has an upper bound represented by a constant $$M$$, i.e., $$|$X^{low}[n]$| \leq M$$ for all $$n$$, we have:

$$
|X^{low}[k]| \leq M \cdot \sqrt{\sum_{n=0}^{N-1} |e^{-j\frac{2\pi}{N}kn}|^2}
$$

Notice that $$|e^{-j\frac{2\pi}{N}kn}|^2$$ is the square of a sine function. This integral can be bounded by a constant less than 1, i.e., $$|e^{-j\frac{2\pi}{N}kn}|^2 \leq 1$$.

$$
|X^{low}[k]| \leq M \cdot \sqrt{\sum_{n=0}^{N-1} 1}
$$

$$
|X^{low}[k]| \leq M \cdot \sqrt{N}
$$, as $$R^{2} \propto N$$,
$$
|X^{low}[k]| \leq M \cdot R^{2}
$$.
