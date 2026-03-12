import copy
import math
import unittest

import torch
from torch import nn

from local_lora.adapters import GroupLocalLoRALinear, InputLocalLoRALinear, LoRALinear
from local_lora.inject import inject_adapters, inject_bd_lora


class TestAdapters(unittest.TestCase):
    def test_scaling_modes(self):
        base = nn.Linear(8, 12, bias=False)
        lora_std = LoRALinear(base, r=16, alpha=16.0, dropout=0.0, scaling_mode="standard")
        self.assertAlmostEqual(lora_std.scaling, 1.0, places=7)

        base2 = nn.Linear(8, 12, bias=False)
        lora_rs = LoRALinear(base2, r=16, alpha=4.0, dropout=0.0, scaling_mode="rs")
        self.assertAlmostEqual(lora_rs.scaling, 1.0, places=7)

    def test_lora_shapes_2d_3d(self):
        base = nn.Linear(8, 12, bias=True)
        mod = LoRALinear(base, r=4, alpha=4.0, dropout=0.0)

        x2 = torch.randn(5, 8)
        y2 = mod(x2)
        self.assertEqual(tuple(y2.shape), (5, 12))
        self.assertEqual(y2.dtype, x2.dtype)

        x3 = torch.randn(2, 7, 8)
        y3 = mod(x3)
        self.assertEqual(tuple(y3.shape), (2, 7, 12))
        self.assertEqual(y3.dtype, x3.dtype)

    def test_group_local_shapes_2d_3d(self):
        base = nn.Linear(8, 12, bias=False)
        mod = GroupLocalLoRALinear(base, r=4, m=2, alpha=4.0, dropout=0.0)

        x2 = torch.randn(5, 8)
        y2 = mod(x2)
        self.assertEqual(tuple(y2.shape), (5, 12))

        x3 = torch.randn(2, 7, 8)
        y3 = mod(x3)
        self.assertEqual(tuple(y3.shape), (2, 7, 12))

    def test_input_local_shapes_2d_3d(self):
        base = nn.Linear(8, 12, bias=False)
        mod = InputLocalLoRALinear(base, r=4, m=2, alpha=4.0, dropout=0.0)

        x2 = torch.randn(5, 8)
        y2 = mod(x2)
        self.assertEqual(tuple(y2.shape), (5, 12))

        x3 = torch.randn(2, 7, 8)
        y3 = mod(x3)
        self.assertEqual(tuple(y3.shape), (2, 7, 12))

    def test_group_local_equals_vanilla_when_g1(self):
        base1 = nn.Linear(8, 12, bias=True)
        base2 = copy.deepcopy(base1)

        lora = LoRALinear(base1, r=4, alpha=4.0, dropout=0.0)
        gl = GroupLocalLoRALinear(base2, r=4, m=4, alpha=4.0, dropout=0.0)  # g=1

        with torch.no_grad():
            gl.lora_A.copy_(lora.lora_A)
            gl.lora_B_grouped[0].copy_(lora.lora_B)

        x = torch.randn(3, 8)
        y_lora = lora(x)
        y_gl = gl(x)
        torch.testing.assert_close(y_lora, y_gl, rtol=1e-6, atol=1e-6)

    def test_input_local_equals_vanilla_when_g1(self):
        base1 = nn.Linear(8, 12, bias=True)
        base2 = copy.deepcopy(base1)

        lora = LoRALinear(base1, r=4, alpha=4.0, dropout=0.0)
        il = InputLocalLoRALinear(base2, r=4, m=4, alpha=4.0, dropout=0.0)  # g=1

        with torch.no_grad():
            il.lora_A_grouped[0].copy_(lora.lora_A)
            il.lora_B.copy_(lora.lora_B)

        x = torch.randn(3, 8)
        y_lora = lora(x)
        y_il = il(x)
        torch.testing.assert_close(y_lora, y_il, rtol=1e-6, atol=1e-6)

    def test_group_local_matches_dense_block_diagonal(self):
        base = nn.Linear(8, 12, bias=False)
        gl = GroupLocalLoRALinear(base, r=4, m=2, alpha=4.0, dropout=0.0)  # g=2, d_block=6

        x = torch.randn(3, 8)
        with torch.no_grad():
            gl.lora_A.uniform_(-0.1, 0.1)
            gl.lora_B_grouped.uniform_(-0.1, 0.1)

        z = x @ gl.lora_A.t()  # (3, 4)
        z_g = z.reshape(3, gl.g, gl.m)  # (3, 2, 2)
        grouped = torch.einsum("bgm,gdm->bgd", z_g, gl.lora_B_grouped).reshape(3, 12)

        # Build dense block-diagonal B: (d_out, r)
        b_dense = torch.zeros((12, 4), dtype=grouped.dtype)
        b_dense[:6, :2] = gl.lora_B_grouped[0]
        b_dense[6:, 2:] = gl.lora_B_grouped[1]
        dense = z @ b_dense.t()

        torch.testing.assert_close(grouped, dense, rtol=1e-6, atol=1e-6)

    def test_input_local_matches_dense_block_diagonal_a(self):
        base = nn.Linear(8, 12, bias=False)
        il = InputLocalLoRALinear(base, r=4, m=2, alpha=4.0, dropout=0.0)  # g=2, d_in_block=4

        x = torch.randn(3, 8)
        with torch.no_grad():
            il.lora_A_grouped.uniform_(-0.1, 0.1)
            il.lora_B.uniform_(-0.1, 0.1)

        x_blocks = x.reshape(3, il.g, il.d_in_block)
        z_blocks = torch.einsum("bgd,gmd->bgm", x_blocks, il.lora_A_grouped).reshape(3, il.r)
        grouped = z_blocks @ il.lora_B.t()

        # Build dense block-diagonal A: (r, d_in)
        a_dense = torch.zeros((il.r, il.in_features), dtype=grouped.dtype)
        a_dense[:2, :4] = il.lora_A_grouped[0]
        a_dense[2:, 4:] = il.lora_A_grouped[1]
        dense = (x @ a_dense.t()) @ il.lora_B.t()

        torch.testing.assert_close(grouped, dense, rtol=1e-6, atol=1e-6)

    def test_constraints(self):
        base = nn.Linear(8, 13, bias=False)
        with self.assertRaises(ValueError):
            GroupLocalLoRALinear(base, r=4, m=3)
        with self.assertRaises(ValueError):
            GroupLocalLoRALinear(base, r=4, m=2)  # g=2, 13 % 2 != 0
        base_in = nn.Linear(13, 8, bias=False)
        with self.assertRaises(ValueError):
            InputLocalLoRALinear(base_in, r=4, m=2)  # g=2, 13 % 2 != 0

    def test_param_counts(self):
        base = nn.Linear(8, 12, bias=False)
        lora = LoRALinear(copy.deepcopy(base), r=4, alpha=4.0, dropout=0.0)
        gl = GroupLocalLoRALinear(copy.deepcopy(base), r=4, m=2, alpha=4.0, dropout=0.0)
        il = InputLocalLoRALinear(copy.deepcopy(base), r=4, m=2, alpha=4.0, dropout=0.0)

        self.assertEqual(lora.adapter_param_counts().total, 4 * 8 + 12 * 4)
        self.assertEqual(gl.adapter_param_counts().total, 4 * 8 + 12 * 2)
        self.assertEqual(il.adapter_param_counts().total, 4 * 4 + 12 * 4)

    def test_inject_replaces_modules(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 12)
                self.k_proj = nn.Linear(8, 12)
                self.other = nn.Linear(8, 12)

            def forward(self, x):
                return self.q_proj(x) + self.k_proj(x) + self.other(x)

        toy = Toy()
        report = inject_adapters(toy, adapter_type="vanilla_lora", r=4, alpha=4.0, dropout=0.0)
        self.assertEqual(len(report.injected), 2)
        self.assertIsInstance(toy.q_proj, LoRALinear)
        self.assertIsInstance(toy.k_proj, LoRALinear)
        self.assertIsInstance(toy.other, nn.Linear)

    def test_inject_bd_lora_mapping(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(8, 12)
                self.o_proj = nn.Linear(8, 12)
                self.up_proj = nn.Linear(8, 16)
                self.down_proj = nn.Linear(16, 8)

            def forward(self, x):
                raise NotImplementedError

        toy = Toy()
        report = inject_bd_lora(
            toy,
            r=4,
            n=2,
            alpha=4.0,
            dropout=0.0,
            bd_row_factor="block_a",
            target_suffixes=("q_proj", "o_proj", "up_proj", "down_proj"),
        )
        self.assertEqual(len(report.injected), 4)
        self.assertIsInstance(toy.q_proj, GroupLocalLoRALinear)
        self.assertIsInstance(toy.up_proj, GroupLocalLoRALinear)
        self.assertIsInstance(toy.o_proj, InputLocalLoRALinear)
        self.assertIsInstance(toy.down_proj, InputLocalLoRALinear)

        toy2 = Toy()
        report2 = inject_bd_lora(
            toy2,
            r=4,
            n=2,
            alpha=4.0,
            dropout=0.0,
            bd_row_factor="dense",
            target_suffixes=("q_proj", "o_proj", "up_proj", "down_proj"),
        )
        self.assertEqual(len(report2.injected), 4)
        self.assertIsInstance(toy2.q_proj, GroupLocalLoRALinear)
        self.assertIsInstance(toy2.up_proj, GroupLocalLoRALinear)
        self.assertIsInstance(toy2.o_proj, LoRALinear)
        self.assertIsInstance(toy2.down_proj, LoRALinear)


if __name__ == "__main__":
    unittest.main()
