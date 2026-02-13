import dataclasses
from typing import Iterator
import itertools
import cuequivariance as cue
import numpy as np
from e3nn import o3


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        self.layout_str = self.layout
        self.layout = getattr(cue, self.layout)
        self.group = O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)


class O3_e3nn(cue.O3):
    def __mul__(  # pylint: disable=no-self-argument
        rep1: "O3_e3nn", rep2: "O3_e3nn"
    ) -> Iterator["O3_e3nn"]:
        return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

    @classmethod
    def clebsch_gordan(
        cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
    ) -> np.ndarray:
        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

        if rep1.p * rep2.p == rep3.p:
            return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                rep3.dim
            )
        return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    def __lt__(  # pylint: disable=no-self-argument
        rep1: "O3_e3nn", rep2: "O3_e3nn"
    ) -> bool:
        rep2 = rep1._from(rep2)
        return (rep1.l, rep1.p) < (rep2.l, rep2.p)

    @classmethod
    def iterator(cls) -> Iterator["O3_e3nn"]:
        for ell in itertools.count(0):
            yield O3_e3nn(l=ell, p=1 * (-1) ** ell)
            yield O3_e3nn(l=ell, p=-1 * (-1) ** ell)
