use super::dag::{OpType, DAG};
use curve25519_dalek::scalar::Scalar;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Core types
// =============================================================================

/// Sparse R1CS matrix in COO (Coordinate) format.
/// Each entry is `(row_index, col_index, scalar_bytes_le)`.
pub type R1CSMatrix = Vec<(usize, usize, [u8; 32])>;

/// Errors from the R1CS witness solvers.
#[derive(Debug)]
pub enum WitnessError {
    /// `circuit_inputs` length doesn't match `num_inputs - num_outputs`.
    InvalidInputs { expected: usize, got: usize },
    /// Fixed-point iteration stalled — circuit is underconstrained or malformed.
    SolverStuck { solved: usize, total: usize },
    /// Row `row` has more than one unknown variable in the forward pass.
    /// The circuit rows are not in topological evaluation order.
    NotInEvaluationOrder { row: usize },
    /// Row `row` defines its single unknown non-linearly: the unknown appears
    /// with non-zero coefficient in **both** A and B, so the constraint is
    /// quadratic in that variable and does not pin a unique value. A circuit
    /// containing such a row is not a function of its inputs.
    NonLinearUnknown { row: usize },
    /// The R1CS instance is structurally malformed or out of range. Every field
    /// of a prover-supplied C* is untrusted, so dimension inconsistencies,
    /// out-of-range row/col indices, or non-canonical scalars are rejected here
    /// rather than panicking during indexing/allocation downstream.
    MalformedInstance { reason: &'static str },
}

/// Sparse R1CS instance for libspartan.
///
/// z-vector layout: `[private_vars(0..num_vars-1) | 1 | outputs... | inputs...]`
///   - `z[0..num_vars-1]`  — private intermediate variables (hidden from verifier)
///   - `z[num_vars]`        — constant 1 (auto-inserted by libspartan)
///   - `z[num_vars+1..]`   — public I/O: `[outputs..., circuit_inputs...]`
///
/// Each constraint row `i`: `<A[i], z> * <B[i], z> = <C[i], z>`
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(non_snake_case)]
pub struct SpartanInstance {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub A: R1CSMatrix,
    pub B: R1CSMatrix,
    pub C: R1CSMatrix,
}

// =============================================================================
// Column assignment (shared by dag_to_spartan and compute_witness)
// =============================================================================

pub(crate) struct ColumnAssignment {
    pub node_to_col: HashMap<usize, usize>,
    pub pow5_intermediates: HashMap<usize, (usize, usize, usize)>,
    pub num_private_vars: usize,
    pub num_public_inputs: usize,
    pub col_const_one: usize,
    pub output_node_order: Vec<usize>,
    pub input_node_order: Vec<usize>,
}

/// Assigns z-vector column indices to all DAG nodes.
///
/// Layout:
///   `0..num_private-1`  → private intermediate variables
///   `num_private`        → constant 1
///   `num_private+1..`   → public I/O [outputs..., inputs...]
pub(crate) fn assign_columns(dag: &DAG) -> ColumnAssignment {
    let output_node_order: Vec<usize> = (0..dag.num_outputs).collect();
    let output_set: HashSet<usize> = output_node_order.iter().cloned().collect();

    let input_node_order: Vec<usize> = dag
        .nodes
        .iter()
        .filter(|n| n.is_input() && !output_set.contains(&n.id))
        .map(|n| n.id)
        .collect();

    let public_set: HashSet<usize> = output_node_order
        .iter()
        .chain(input_node_order.iter())
        .cloned()
        .collect();

    let mut node_to_col: HashMap<usize, usize> = HashMap::new();
    let mut pow5_intermediates: HashMap<usize, (usize, usize, usize)> = HashMap::new();
    let mut next_private = 0usize;

    for node in &dag.nodes {
        if !public_set.contains(&node.id) {
            node_to_col.insert(node.id, next_private);
            next_private += 1;
        }
        if let OpType::Pow5(src) = node.op {
            let sq_col = next_private;
            next_private += 1;
            let qd_col = next_private;
            next_private += 1;
            pow5_intermediates.insert(node.id, (sq_col, qd_col, src));
        }
    }

    let num_private_vars = next_private;
    let col_const_one = num_private_vars;

    let num_public_inputs = output_node_order.len() + input_node_order.len();
    let mut public_offset = 0usize;
    for &node_id in output_node_order.iter().chain(input_node_order.iter()) {
        node_to_col.insert(node_id, num_private_vars + 1 + public_offset);
        public_offset += 1;
    }

    ColumnAssignment {
        node_to_col,
        pow5_intermediates,
        num_private_vars,
        num_public_inputs,
        col_const_one,
        output_node_order,
        input_node_order,
    }
}

#[inline]
fn push_entry(matrix: &mut R1CSMatrix, row: usize, col: usize, val: Scalar) {
    if val != Scalar::ZERO {
        matrix.push((row, col, val.to_bytes()));
    }
}

// =============================================================================
// dag_to_spartan
// =============================================================================

/// Converts a DAG to Spartan R1CS matrices.
///
/// Rows are emitted in **topological evaluation order** (reverse node-ID order):
/// deep dependencies first, output constraints last. This guarantees that a
/// single forward pass can compute the witness.
pub fn dag_to_spartan(dag: &DAG) -> SpartanInstance {
    let cols = assign_columns(dag);

    let mut a_mat: R1CSMatrix = Vec::new();
    let mut b_mat: R1CSMatrix = Vec::new();
    let mut c_mat: R1CSMatrix = Vec::new();
    let mut row = 0usize;

    for node in dag.nodes.iter().rev() {
        match node.op {
            OpType::Input => {}

            OpType::Alias(src) => {
                // node * 1 = src
                let node_col = cols.node_to_col[&node.id];
                let src_col = cols.node_to_col[&src];
                push_entry(&mut a_mat, row, node_col, Scalar::ONE);
                push_entry(&mut b_mat, row, cols.col_const_one, Scalar::ONE);
                push_entry(&mut c_mat, row, src_col, Scalar::ONE);
                row += 1;
            }

            OpType::Add(l, r) => {
                // (l + r) * 1 = node
                let l_col = cols.node_to_col[&l];
                let r_col = cols.node_to_col[&r];
                let out_col = cols.node_to_col[&node.id];
                push_entry(&mut a_mat, row, l_col, Scalar::ONE);
                push_entry(&mut a_mat, row, r_col, Scalar::ONE);
                push_entry(&mut b_mat, row, cols.col_const_one, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Mul(l, r) => {
                // l * r = node
                let l_col = cols.node_to_col[&l];
                let r_col = cols.node_to_col[&r];
                let out_col = cols.node_to_col[&node.id];
                push_entry(&mut a_mat, row, l_col, Scalar::ONE);
                push_entry(&mut b_mat, row, r_col, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Scale(src, k) => {
                // (k * src) * 1 = node
                let src_col = cols.node_to_col[&src];
                let out_col = cols.node_to_col[&node.id];
                push_entry(&mut a_mat, row, src_col, Scalar::from(k));
                push_entry(&mut b_mat, row, cols.col_const_one, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Pow5(_) => {
                // x^5 unrolled: sq = src*src, qd = sq*sq, out = qd*src
                let &(sq_col, qd_col, src_id) = cols.pow5_intermediates.get(&node.id).unwrap();
                let src_col = cols.node_to_col[&src_id];
                let out_col = cols.node_to_col[&node.id];

                // src * src = sq
                push_entry(&mut a_mat, row, src_col, Scalar::ONE);
                push_entry(&mut b_mat, row, src_col, Scalar::ONE);
                push_entry(&mut c_mat, row, sq_col, Scalar::ONE);
                row += 1;

                // sq * sq = qd
                push_entry(&mut a_mat, row, sq_col, Scalar::ONE);
                push_entry(&mut b_mat, row, sq_col, Scalar::ONE);
                push_entry(&mut c_mat, row, qd_col, Scalar::ONE);
                row += 1;

                // qd * src = out
                push_entry(&mut a_mat, row, qd_col, Scalar::ONE);
                push_entry(&mut b_mat, row, src_col, Scalar::ONE);
                push_entry(&mut c_mat, row, out_col, Scalar::ONE);
                row += 1;
            }

            OpType::Output | OpType::Undefined => {}
        }
    }

    SpartanInstance {
        num_cons: row,
        num_vars: cols.num_private_vars,
        num_inputs: cols.num_public_inputs,
        num_outputs: cols.output_node_order.len(),
        A: a_mat,
        B: b_mat,
        C: c_mat,
    }
}

// =============================================================================
// compute_witness (DAG-based, used for C0)
// =============================================================================

/// Computes the full witness from the DAG directly.
///
/// Returns `(vars, public_io)`:
/// - `vars`: private variable values, `len = num_vars`
/// - `public_io`: `[outputs..., circuit_inputs...]`, `len = num_inputs`
pub fn compute_witness(dag: &DAG, input_values: &[Scalar]) -> (Vec<Scalar>, Vec<Scalar>) {
    let cols = assign_columns(dag);

    assert_eq!(
        input_values.len(),
        cols.input_node_order.len(),
        "Expected {} input values, got {}",
        cols.input_node_order.len(),
        input_values.len()
    );

    let mut node_values: Vec<Option<Scalar>> = vec![None; dag.nodes.len()];

    for (i, &node_id) in cols.input_node_order.iter().enumerate() {
        node_values[node_id] = Some(input_values[i]);
    }

    for node in dag.nodes.iter().rev() {
        match node.op {
            OpType::Input => {}
            OpType::Add(l, r) => {
                node_values[node.id] = Some(node_values[l].unwrap() + node_values[r].unwrap());
            }
            OpType::Mul(l, r) => {
                node_values[node.id] = Some(node_values[l].unwrap() * node_values[r].unwrap());
            }
            OpType::Alias(src) => {
                node_values[node.id] = node_values[src];
            }
            OpType::Scale(src, k) => {
                node_values[node.id] = Some(Scalar::from(k) * node_values[src].unwrap());
            }
            OpType::Pow5(src) => {
                let x = node_values[src].unwrap();
                let sq = x * x;
                node_values[node.id] = Some(sq * sq * x);
            }
            _ => {}
        }
    }

    let mut vars = vec![Scalar::ZERO; cols.num_private_vars];
    for (&node_id, &col) in &cols.node_to_col {
        if col < cols.num_private_vars {
            vars[col] = node_values[node_id].expect("private node value not computed");
        }
    }
    for (_, &(sq_col, qd_col, src_id)) in &cols.pow5_intermediates {
        let x = node_values[src_id].unwrap();
        let sq = x * x;
        vars[sq_col] = sq;
        vars[qd_col] = sq * sq;
    }

    let mut public_io = Vec::with_capacity(cols.num_public_inputs);
    for &node_id in &cols.output_node_order {
        public_io.push(node_values[node_id].expect("output node value not computed"));
    }
    for &node_id in &cols.input_node_order {
        public_io.push(node_values[node_id].unwrap());
    }

    (vars, public_io)
}

// =============================================================================
// solve_witness_forward (single-pass, used for C*)
// =============================================================================

/// Computes the witness using a **single forward pass** over R1CS rows.
///
/// Requires rows to be in **topological evaluation order**: when row `i` is
/// reached, all variables in A and B must already be known except at most one.
/// Violations return `Err(WitnessError::NotInEvaluationOrder { row })`.
///
/// O(n) — no backtracking.
pub fn solve_witness_forward(
    instance: &SpartanInstance,
    num_outputs: usize,
    circuit_inputs: &[Scalar],
) -> Result<(Vec<Scalar>, Vec<Scalar>), WitnessError> {
    // Validate the untrusted C* before the `num_inputs - num_outputs`
    // subtraction below and before any allocation/indexing.
    validate_instance(instance)?;
    let expected = instance.num_inputs - num_outputs;
    if circuit_inputs.len() != expected {
        return Err(WitnessError::InvalidInputs {
            expected,
            got: circuit_inputs.len(),
        });
    }

    let (a_rows, b_rows, c_rows) = build_row_views(instance);

    let z_len = instance.num_vars + 1 + instance.num_inputs;
    let mut z = vec![Scalar::ZERO; z_len];
    let mut solved = vec![false; z_len];

    z[instance.num_vars] = Scalar::ONE;
    solved[instance.num_vars] = true;
    for (i, &val) in circuit_inputs.iter().enumerate() {
        let idx = instance.num_vars + 1 + num_outputs + i;
        z[idx] = val;
        solved[idx] = true;
    }

    for row in 0..instance.num_cons {
        let mut unsolved_col: Option<usize> = None;
        let mut multi = false;

        for &(col, _) in a_rows[row]
            .iter()
            .chain(b_rows[row].iter())
            .chain(c_rows[row].iter())
        {
            if !solved[col] {
                match unsolved_col {
                    None => unsolved_col = Some(col),
                    Some(prev) if prev == col => {}
                    Some(_) => {
                        multi = true;
                        break;
                    }
                }
            }
        }

        if multi {
            return Err(WitnessError::NotInEvaluationOrder { row });
        }

        let j = match unsolved_col {
            Some(j) => j,
            None => continue,
        };

        let (a_known, a_j) = accumulate(&a_rows[row], j, &z);
        let (b_known, b_j) = accumulate(&b_rows[row], j, &z);
        let (c_known, c_j) = accumulate(&c_rows[row], j, &z);

        // A defining row must be linear in its single unknown: the unknown may
        // appear in at most one of {A, B} (it may also appear in C). If it
        // appears in both A and B the constraint is quadratic in z[j] — it does
        // not pin a unique value, so C* would not be a function, and the linear
        // solve below would silently produce an incorrect value. Reject it.
        if a_j != Scalar::ZERO && b_j != Scalar::ZERO {
            return Err(WitnessError::NonLinearUnknown { row });
        }

        let denom = a_j * b_known + b_j * a_known - c_j;
        if denom == Scalar::ZERO {
            continue;
        }
        z[j] = (c_known - a_known * b_known) * denom.invert();
        solved[j] = true;
    }

    check_convergence(&solved, instance.num_vars, num_outputs, instance.num_vars)?;
    Ok(extract_result(&z, instance))
}

// =============================================================================
// Helpers
// =============================================================================

/// Validates an untrusted `SpartanInstance` before any indexing or allocation
/// derived from its fields.
///
/// Every field of a prover-supplied C* is attacker-controlled, so the cheap
/// structural checks below must hold before `build_row_views` / `accumulate` /
/// `dot` can index safely — otherwise a malformed C* panics or OOMs the
/// verifier instead of being rejected. Called at the entry of
/// [`solve_witness_forward`] and [`satisfies`].
///
/// `num_cons` itself is intentionally NOT bounded here: the verifier bounds it
/// via its `K* < K0` check before the function-check ever runs.
pub(crate) fn validate_instance(instance: &SpartanInstance) -> Result<(), WitnessError> {
    // Guards the `num_inputs - num_outputs` subtraction performed in the solver
    // and in the verifier.
    if instance.num_inputs < instance.num_outputs {
        return Err(WitnessError::MalformedInstance {
            reason: "num_inputs < num_outputs",
        });
    }
    // In a triangular R1CS every private variable needs its own defining row,
    // so num_vars <= num_cons holds for any circuit that could pass the
    // function-check. This never rejects a valid circuit and bounds the z
    // allocation (num_cons is itself bounded by the verifier's K* < K0 check).
    if instance.num_vars > instance.num_cons {
        return Err(WitnessError::MalformedInstance {
            reason: "num_vars > num_cons",
        });
    }
    // Checked arithmetic so the z-vector length cannot overflow into a giant
    // allocation (which would abort/OOM instead of erroring).
    let z_len = instance
        .num_vars
        .checked_add(1)
        .and_then(|x| x.checked_add(instance.num_inputs))
        .ok_or(WitnessError::MalformedInstance {
            reason: "dimension overflow",
        })?;

    // Every (row, col, scalar) must be in range and canonical. After this pass,
    // the `.unwrap()`s and `a_rows[row]` indexing in `build_row_views` are sound.
    for &(row, col, bytes) in instance
        .A
        .iter()
        .chain(instance.B.iter())
        .chain(instance.C.iter())
    {
        if row >= instance.num_cons
            || col >= z_len
            || !bool::from(Scalar::from_canonical_bytes(bytes).is_some())
        {
            return Err(WitnessError::MalformedInstance {
                reason: "row/col out of range or non-canonical scalar",
            });
        }
    }
    Ok(())
}

fn build_row_views(
    instance: &SpartanInstance,
) -> (
    Vec<Vec<(usize, Scalar)>>,
    Vec<Vec<(usize, Scalar)>>,
    Vec<Vec<(usize, Scalar)>>,
) {
    // Precondition: `validate_instance(instance)` has already run, so every row
    // index is < num_cons and every scalar is canonical — the `a_rows[row]`
    // indexing below and the `from_canonical_bytes(...).unwrap()` cannot panic.
    let mut a_rows = vec![Vec::new(); instance.num_cons];
    let mut b_rows = vec![Vec::new(); instance.num_cons];
    let mut c_rows = vec![Vec::new(); instance.num_cons];

    for &(row, col, bytes) in &instance.A {
        a_rows[row].push((col, Scalar::from_canonical_bytes(bytes).unwrap()));
    }
    for &(row, col, bytes) in &instance.B {
        b_rows[row].push((col, Scalar::from_canonical_bytes(bytes).unwrap()));
    }
    for &(row, col, bytes) in &instance.C {
        c_rows[row].push((col, Scalar::from_canonical_bytes(bytes).unwrap()));
    }

    (a_rows, b_rows, c_rows)
}

/// Splits a row into `(known_sum, coeff_of_j)`.
fn accumulate(row: &[(usize, Scalar)], j: usize, z: &[Scalar]) -> (Scalar, Scalar) {
    let mut known = Scalar::ZERO;
    let mut coeff_j = Scalar::ZERO;
    for &(col, val) in row {
        if col == j {
            coeff_j += val;
        } else {
            known += val * z[col];
        }
    }
    (known, coeff_j)
}

fn check_convergence(
    solved: &[bool],
    num_vars: usize,
    num_outputs: usize,
    const_col: usize,
) -> Result<(), WitnessError> {
    let total = num_vars + num_outputs;
    let mut count = 0;
    for i in 0..num_vars {
        if solved[i] {
            count += 1;
        }
    }
    for i in 0..num_outputs {
        if solved[const_col + 1 + i] {
            count += 1;
        }
    }
    if count < total {
        Err(WitnessError::SolverStuck {
            solved: count,
            total,
        })
    } else {
        Ok(())
    }
}

fn extract_result(z: &[Scalar], instance: &SpartanInstance) -> (Vec<Scalar>, Vec<Scalar>) {
    let vars = z[..instance.num_vars].to_vec();
    let public_io = z[instance.num_vars + 1..instance.num_vars + 1 + instance.num_inputs].to_vec();
    (vars, public_io)
}

/// Inner product of a sparse row with the assignment `z`.
fn dot(row: &[(usize, Scalar)], z: &[Scalar]) -> Scalar {
    let mut acc = Scalar::ZERO;
    for &(col, val) in row {
        acc += val * z[col];
    }
    acc
}

/// Checks that a full assignment satisfies every R1CS constraint:
/// `<A[i],z> * <B[i],z> = <C[i],z>` for all rows.
///
/// `z` layout: `[private_vars(0..num_vars-1) | 1 | public_io(0..num_inputs-1)]`.
///
/// Used by the verifier to confirm the forward-solved witness for C* actually
/// satisfies every constraint. Combined with triangularizability (enforced by
/// [`solve_witness_forward`]), a self-consistent assignment implies C* has a
/// unique witness per input — i.e. C* is a function — independent of π*.
pub fn satisfies(instance: &SpartanInstance, vars: &[Scalar], public_io: &[Scalar]) -> bool {
    if validate_instance(instance).is_err() {
        return false;
    }
    if vars.len() != instance.num_vars || public_io.len() != instance.num_inputs {
        return false;
    }
    let (a_rows, b_rows, c_rows) = build_row_views(instance);

    let mut z = vec![Scalar::ZERO; instance.num_vars + 1 + instance.num_inputs];
    z[..instance.num_vars].copy_from_slice(vars);
    z[instance.num_vars] = Scalar::ONE;
    z[instance.num_vars + 1..].copy_from_slice(public_io);

    for row in 0..instance.num_cons {
        if dot(&a_rows[row], &z) * dot(&b_rows[row], &z) != dot(&c_rows[row], &z) {
            return false;
        }
    }
    true
}
