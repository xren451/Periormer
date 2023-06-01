import torch

def fftTransfer1(timeseries, per_term, fmin=0.2):
    import pandas as pd
    import numpy as np
    import math
    from scipy.fftpack import fft, ifft
    import matplotlib.pyplot as plt
    import seaborn
    import scipy.signal as signal


    yf = abs(fft(timeseries))  # 取绝对值
    yfnormlize = yf / len(timeseries)  # 归一化处理
    yfhalf = yfnormlize[range(int(len(timeseries) / 2))]  # 由于对称性，只取一半区间
    yfhalf = yfhalf * 2  # y 归一化

    xf = np.arange(len(timeseries))  # 频率
    xhalf = xf[range(int(len(timeseries) / 2))]  # 取一半区间

    #     plt.subplot(212)
    #     plt.plot(xhalf, yfhalf, 'r')
    #     plt.title('FFT of Mixed wave(half side frequency range)', fontsize=10, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表

    fwbest = yfhalf[signal.argrelextrema(yfhalf, np.greater)]  # Amplitude
    xwbest = signal.argrelextrema(yfhalf, np.greater)  # Frequency
    #     plt.plot(xwbest[0][:n], fwbest[:n], 'o', c='yellow')
    #     plt.show(block=False)
    #     plt.show()

    xorder = np.argsort(-fwbest)  # 对获取到的极值进行降序排序，也就是频率越接近，越排前
    #print('xorder = ', xorder)
    xworder = list()
    xworder.append(xwbest[x] for x in xorder)  # 返回频率从大到小的极值顺序
    fworder = list()
    fworder.append(fwbest[x] for x in xorder)  # 返回幅度
    fwbest = fwbest[fwbest >= fmin].copy()
    x=len(timeseries) / xwbest[0][:len(fwbest)]
    y=fwbest
    a=np.zeros((len(y),2), dtype='float32')
    a[:,0]=y#Get amplitude
    a[:,1]=x#Get Periodic terms
    df=pd.DataFrame(a)
    df.set_axis(["amp","period"],axis=1,inplace=True)
    # sorting data frame by name
    df.sort_values("amp", axis = 0, ascending = False,
                 inplace = True, na_position ='last')
    df=df.iloc[:int(per_term),:]
    return df

# def fftTransfer3D(timeseries, per_term, fmin=0.2):
#     import pandas as pd
#     import numpy as np
#     import math
#     from scipy.fftpack import fft, ifft
#     import matplotlib.pyplot as plt
#     import seaborn
#     import scipy.signal as signal
#     from models.Period import fftTransfer1
#     outputFFT=[]
#     y=timeseries
#     for i in range(1,y.shape[1]):
# #        y=pd.read_csv("ETTh1.csv")
#         y=pd.DataFrame(y)
#         y=y.iloc[:,i];
#         y=y.values
#         x=fftTransfer1(y,per_term,fmin = 0.015)
#         outputFFT.append(x)
#     outputFFT=np.array(outputFFT)#(feature*n*FFTout_channel)
#     return outputFFT
#
# import numpy as np
# import pandas as pd
# y=pd.read_csv("ETTh1.csv")
# outputFFT=[]
# for i in range(1,y.shape[1]):
#     y=pd.read_csv("ETTh1.csv")
#     y=pd.DataFrame(y)
#     y=y.iloc[:,i];
#     y=y.values
#     x=fftTransfer1(y,per_term,fmin = 0.015)
#     outputFFT.append(x)
# outputFFT=np.array(outputFFT)#(feature*n*FFTout_channel)

def get_data():
    import torch
    import math
    t = torch.linspace(0., 4 * math.pi, 100)

    start = torch.rand(128) * 2 * math.pi
    x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos[:64] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
    x_pos += 0.01 * torch.randn_like(x_pos)
    y_pos += 0.01 * torch.randn_like(y_pos)
    ######################
    # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
    # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
    ######################
    X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos,x_pos, y_pos], dim=2)
    y = torch.zeros(128)
    y[:64] = 1

    perm = torch.randperm(128)
    X = X[perm]
    y = y[perm]

    ######################
    # t is a tensor of times of shape (sequence=100,)
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise respectively.
    ######################
    return t, X, y

class VectorField(torch.nn.Module):

    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(z)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out

def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
    import torchdiffeq
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """

    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    vector_field = func(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
                         "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
                         " channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))
    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out