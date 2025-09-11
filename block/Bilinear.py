# Auto-generated single-file for Bilinear
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch import Tensor
import math

# ---- original imports from contributing modules ----
from torch.nn import init

# ---- Bilinear (target) ----
class Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.

    Args:
        in1_features: size of each first input sample, must be > 0
        in2_features: size of each second input sample, must be > 0
        out_features: size of each output sample, must be > 0
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(*, H_\text{in1})` where :math:`H_\text{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_\text{in2})` where :math:`H_\text{in2}=\text{in2\_features}`.
        - Output: :math:`(*, H_\text{out})` where :math:`H_\text{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """

    __constants__ = ["in1_features", "in2_features", "out_features"]
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if in1_features <= 0:
            raise ValueError(f"in1_features must be > 0, but got {in1_features}")
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in1_features, in2_features), **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )
