// Copyright 2026 Eric Kryski (@ekryski) and Tom Turney (@TheTom)
// SPDX-License-Identifier: Apache-2.0
//! End-to-end smoke for the ffai-vulkan Device: drive a real kernel through
//! the `ffai_core::Device` seam (upload → alloc → dispatch → download) on a
//! live Vulkan device and compare to a CPU oracle. Mirrors metaltile's
//! `vulkan_smoke.rs` but exercises the ffai-core trait, not run_kernel
//! directly. Skips cleanly if no Vulkan device is present.
#![cfg(feature = "vulkan")]

use ffai_core::{Binding, Grid};
use ffai_vulkan::VulkanDevice;
use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, IndexExpr, Kernel, Op, Param, ParamKind, ValueId},
    shape::Shape,
};

fn vector_add_ir() -> Kernel {
    let mut k = Kernel::new("vector_add");
    for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
        k.params.push(Param {
            name: name.into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: is_out,
            kind: ParamKind::Tensor,
        });
    }
    k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
    k.body.name_value(ValueId::new(0), "idx");
    k.body.push_op(
        Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
        ValueId::new(1),
    );
    k.body.name_value(ValueId::new(1), "x");
    k.body.push_op(
        Op::Load { src: "b".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
        ValueId::new(2),
    );
    k.body.name_value(ValueId::new(2), "y");
    k.body.push_op(
        Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
        ValueId::new(3),
    );
    k.body.name_value(ValueId::new(3), "sum");
    k.body.push_op_no_result(Op::Store {
        dst: "c".into(),
        indices: vec![IndexExpr::Value(ValueId::new(0))],
        value: ValueId::new(3),
        mask: None,
    });
    k
}

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn ffai_vulkan_vector_add_f32_bit_exact() {
    let Some(dev) = VulkanDevice::create().expect("vulkan create") else {
        eprintln!("ffai_vulkan_smoke: no Vulkan device — skipping");
        return;
    };
    eprintln!("ffai_vulkan_smoke: device='{}'", dev.name());

    const N: usize = 8 * 1024;
    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32) * -0.25 + 7.0).collect();
    let oracle: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let ba = dev.upload(&to_bytes(&a)).unwrap();
    let bb = dev.upload(&to_bytes(&b)).unwrap();
    let bc = dev.alloc(N * 4).unwrap();

    let block = 256u32;
    let grid = (N as u32).div_ceil(block);
    let bindings = [Binding::Buffer(ba), Binding::Buffer(bb), Binding::Buffer(bc.clone())];
    dev.dispatch(&vector_add_ir(), &bindings, Grid::d1(grid, block))
        .expect("dispatch");
    dev.synchronize().unwrap();

    let mut out = vec![0u8; N * 4];
    dev.download(bc.as_ref(), &mut out).unwrap();
    let c: Vec<f32> = out
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();

    let max_abs = c
        .iter()
        .zip(&oracle)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    eprintln!("ffai_vulkan_smoke: max|Δ| = {max_abs:e}");
    assert!(max_abs == 0.0, "vector_add not bit-exact through ffai seam: {max_abs:e}");
}
